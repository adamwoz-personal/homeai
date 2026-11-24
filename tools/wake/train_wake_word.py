#!/usr/bin/env python3
"""Train a custom openWakeWord model ("hey jeeves"; plans/WAKE_WORD_PLAN.md).

Why: Adam wants the assistant called Jeeves (2026-11-23) and no Jeeves model exists. openWakeWord's own trainer
(installed by tools/wake/install_wake_train.sh) runs three steps: generate synthetic clips of the phrase and of
sound-alike phrases with Piper's LibriTTS-R generator (hundreds of voices), mix them with room echoes and
background sound, then train a small model against ~2000 h of negative features, stopping on a false-positive
target. This writes the config and runs the steps in the training venv, with the coder paused (the card is full
with Jarvis and the coder; Jarvis stays on).

Sound-alikes ("hey jarvis", "jesus", "sleeves", "he leaves"...) are given as adversarial negatives so the model
learns they are not the name.

Usage:
    .venv/bin/python tools/wake/train_wake_word.py [--phrase "hey jeeves"] [--samples 20000] [--steps 50000]
        [--step generate|augment|train|all] [--run NAME]
Output: ~/.cache/homeai/wake-train/runs/<name>/ (clips, features), model copied to ~/homeai-bench/wake-<name>/.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROOT = Path.home() / ".local/share/homeai/wake-train"
DATA = Path.home() / ".cache/homeai/wake-train"
PY = ROOT / "venv/bin/python"
TRAIN = ROOT / "src/openWakeWord/openwakeword/train.py"
PIPER = ROOT / "src/piper-sample-generator"
BENCH = Path.home() / "homeai-bench"

# Words that sound like "jeeves", and the current name, so a model never wakes on them.
SOUND_ALIKES = ["hey jarvis", "jarvis", "hey jesus", "jesus", "hey steve", "hey steven", "hey eve", "cheese",
                "hey cheese", "jeans", "sleeves", "he leaves", "she leaves", "leaves", "thieves", "hey jeep", "jeez",
                "hey geez", "hey reeves", "keanu reeves", "believes", "hey keys", "achieves", "hey cheeks",
                "hey jean", "hey genie", "hey gee", "hey these", "he sees", "heaves", "chives", "hey g's"]


def sound_alike_list(samples: int, share: float = 0.25) -> list[str]:
    """The sound-alikes repeated to be about ``share`` of the negative clips. openWakeWord puts this list in front
    of ~``samples`` generated phrases and voices each text once, so a list given once was ~32 clips in 50,000 and
    the model kept waking on "hey jesus" (39/54 held-out, round 3, 2026-11-23)."""
    return SOUND_ALIKES * max(1, round(samples * share / len(SOUND_ALIKES)))


def config_for(name: str, phrase: str, samples: int, steps: int, out: Path, negative_weight: int = 1500,
               layer: int = 32, rounds: int = 1, adversarial: int = 50, alike_share: float = 0.25) -> dict:
    """openWakeWord's training config. negative_weight: how hard false wakes are punished at the end of training
    (upstream default 1500 gave 0 false/h but 34% recall on held-out voices, 2026-11-23); layer: model width;
    rounds: augmentation passes over each clip; adversarial: sound-alike clips per training batch ("hey jesus"
    woke the 300-weight model on 37 of 54 held-out clips with the default 50); alike_share: the sound-alikes'
    part of the negative clips (0.25 cut "hey jesus" to 10/54 but recall to 59%, round 4)."""
    return {
        "model_name": name,
        "target_phrase": [phrase],
        "custom_negative_phrases": sound_alike_list(samples, alike_share),
        "n_samples": samples,
        "n_samples_val": max(1000, samples // 10),
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(PIPER),
        "output_dir": str(out),
        "rir_paths": [str(DATA / "mit_rirs")],
        "background_paths": [str(DATA / "audioset_16k")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(DATA / "validation_set_features.npy"),
        "augmentation_rounds": rounds,
        "feature_data_files": {"ACAV100M_sample": str(DATA / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": adversarial, "positive": 50},
        "model_type": "dnn",
        "layer_size": layer,
        "steps": steps,
        "max_negative_weight": negative_weight,
        "target_false_positives_per_hour": 0.2,
        "target_accuracy": 0.7,
        "target_recall": 0.5,
    }


def single_file(src: Path, dest: Path) -> Path:
    """The model as one file. PyTorch's new ONNX exporter puts the weights in a separate "<name>.onnx.data";
    copied alone, the .onnx can't load ("External data path does not exist"), so the weights go inside."""
    import onnx
    onnx.save(onnx.load(str(src)), str(dest), save_as_external_data=False)
    return dest


def coder_paused():
    """The coder stopped for the block, voice mode restored after (as the art tools: gpu_mode.coder_paused)."""
    from homeai import gpu_mode as G
    return G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True)).coder_paused(False)


def missing(step: str = "all") -> list[str]:
    """What the step still needs. Generating clips needs only the code and the voice generator, so it can run
    while the 17 GB of negative features are still downloading."""
    need = [PY, TRAIN, PIPER / "generate_samples.py", PIPER / "models/en_US-libritts_r-medium.pt"]
    if step in ("train", "all"):            # only training reads the negative features
        need += [DATA / "validation_set_features.npy", DATA / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"]
    out = [str(p) for p in need if not p.exists()]
    if step != "generate":
        out += [str(d) for d in (DATA / "mit_rirs", DATA / "audioset_16k") if not any(d.glob("*.wav"))]
    return out


FEATURES = ("positive_features_train.npy", "positive_features_test.npy", "negative_features_train.npy",
            "negative_features_test.npy")


def clear_partial_features(folder: Path) -> list[str]:
    """Delete the feature files when only some exist, so augmenting makes all four. openWakeWord skips augmenting
    if positive_features_train.npy exists, so deleting only the negatives (to redo them) left training without
    them and it failed (round 5, 2026-11-24). Returns what was deleted."""
    present = [folder / f for f in FEATURES if (folder / f).exists()]
    if len(present) in (0, len(FEATURES)):
        return []
    for f in present:
        f.unlink()
    return [f.name for f in present]


def run_step(step: str, config: Path, log: Path) -> float:
    start = time.monotonic()
    with log.open("a") as out:
        out.write(f"\n=== {step} {time.strftime('%F %T')}\n")
        out.flush()
        # cwd = the run folder: train.py writes relative temp files
        code = subprocess.call([str(PY), str(TRAIN), "--training_config", str(config), f"--{step}"],
                               cwd=config.parent, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"{step} failed (exit {code}); see {log}")
    return round(time.monotonic() - start, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phrase", default="hey jeeves")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--step", default="all", choices=["generate", "augment", "train", "all"])
    ap.add_argument("--run", default="", help="run name (default: phrase and sample count)")
    ap.add_argument("--negative-weight", type=int, default=1500)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=1, help="augmentation passes (delete old features to redo)")
    ap.add_argument("--adversarial", type=int, default=50, help="sound-alike clips per training batch")
    ap.add_argument("--alike-share", type=float, default=0.25, help="sound-alikes' share of the negative clips")
    ap.add_argument("--tag", default="", help="suffix for the saved copy, to keep several trainings of one run")
    ap.add_argument("--keep-coder", action="store_true", help="don't pause the coder (only if the GPU has room)")
    args = ap.parse_args(argv)
    gone = missing(args.step)
    if gone:
        print("not installed yet (tools/wake/install_wake_train.sh):\n  " + "\n  ".join(gone))
        return 1
    name = args.phrase.replace(" ", "_")
    run = args.run or f"{name}-{args.samples}"
    folder = DATA / "runs" / run
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "config.yml"
    import yaml  # noqa: PLC0415 - only the driver needs it, from the project venv
    config.write_text(yaml.safe_dump(config_for(name, args.phrase, args.samples, args.steps, folder / "model",
                                                args.negative_weight, args.layer, args.rounds, args.adversarial,
                                                args.alike_share)))
    log = folder / "train.log"
    steps = ["generate_clips", "augment_clips", "train_model"] if args.step == "all" else \
        [{"generate": "generate_clips", "augment": "augment_clips", "train": "train_model"}[args.step]]
    times = {}
    pause = contextlib.nullcontext() if args.keep_coder else coder_paused()
    with pause:
        for step in steps:
            if step == "augment_clips":
                gone = clear_partial_features(folder / "model" / name)
                if gone:
                    print("  partial features deleted so all four are made: " + ", ".join(gone), flush=True)
            print(f"{step} ... (log {log})", flush=True)
            times[step] = run_step(step, config, log)
            print(f"{step} done in {times[step]} s", flush=True)
    model = folder / "model" / f"{name}.onnx"
    if model.exists():
        dest = BENCH / f"wake-{run}{'-' + args.tag if args.tag else ''}"
        dest.mkdir(parents=True, exist_ok=True)
        single_file(model, dest / model.name)
        shutil.copy2(config, dest / "config.yml")
        (dest / "times.json").write_text(json.dumps(times, indent=1))
        print(dest / model.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
