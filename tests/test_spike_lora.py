"""The LoRA spike can't take the box down: one trainer, one card, RAM capped below what's free."""
from __future__ import annotations

from pathlib import Path

from tools.art import spike_lora as S


def test_every_step_is_capped_and_sees_only_the_big_card(tmp_path):
    for name, cmd in S.commands("4b", tmp_path / "d.toml", tmp_path, 50):
        full = S.capped(cmd, 16)
        assert full[:4] == ["systemd-run", "--user", "--scope", "-q"]
        assert "MemoryMax=16G" in full and "MemorySwapMax=0" in full
        assert full[full.index("env") + 1] == "HIP_VISIBLE_DEVICES=0"        # the iGPU made two trainers


def test_one_trainer_process(tmp_path):
    train = dict(S.commands("4b", tmp_path / "d.toml", tmp_path, 50))["train"]
    assert train[train.index("--num_processes") + 1] == "1"
    assert train[train.index("--max_train_steps") + 1] == "50"


def test_default_cap_sits_under_free_ram():
    import inspect
    assert "default=16)" in inspect.getsource(S.main)            # 22 was above the ~20 GB free in voice mode


def test_dataset_captions_are_the_plain_words(tmp_path):
    toml = S.dataset(tmp_path, 512).read_text()
    assert "resolution = [512, 512]" in toml
    rows = (tmp_path / "data.jsonl").read_text().splitlines()
    assert len(rows) == len(S.PICTURES) and all("fighting a chimera" in r and "goat" not in r for r in rows)
    assert all(Path(__import__("json").loads(r)["image_path"]).parent == S.BENCH for r in rows)
