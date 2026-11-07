"""The art tools' ComfyUI graphs are wired correctly (tools/art/comfy_common.py). No GPU needed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "art"))

import comfy_common  # noqa: E402


@pytest.mark.parametrize("model", comfy_common.MODELS)
def test_every_link_points_at_a_node(model):
    graph = comfy_common.workflow(model, "a fox", 7)
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def _seed(graph):
    return graph["noise"]["inputs"]["noise_seed"] if "noise" in graph else next(
        n["inputs"].get("seed", n["inputs"].get("noise_seed")) for k, n in graph.items() if k == "sample")


@pytest.mark.parametrize("model", comfy_common.MODELS)
def test_prompt_seed_and_size_reach_the_graph(model):
    graph = comfy_common.workflow(model, "a fox", 7, 640, 480)
    text = graph.get("pos", graph.get("text"))["inputs"]
    assert "a fox" in text.get("text", text.get("prompt"))
    assert _seed(graph) == 7
    scale = 2 if model == "hidream" else 1
    assert (graph["latent"]["inputs"]["width"], graph["latent"]["inputs"]["height"]) == (640 * scale, 480 * scale)


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        comfy_common.workflow("dalle", "a fox", 7)


def test_z_image_gets_its_sampling_shift():
    graph = comfy_common.workflow("zimage", "a fox", 7)
    assert graph["sample"]["inputs"]["model"] == ["shift", 0]


def test_comfy_runs_capped_and_local():
    """It reached 26.8 GB of 29 GB and the OOM killer took Adam's terminal with it (2026-11-05)."""
    command = comfy_common.comfy_command()
    assert command[:3] == ["systemd-run", "--user", "--scope"]
    assert f"MemoryMax={comfy_common.MEMORY_CAP}" in command and "MemorySwapMax=0" in command
    assert command[command.index("--listen") + 1] == "127.0.0.1"



def test_print_grid_puts_a_line_on_every_inch():
    """1 inch squares for tabletop figures: spacing has to be exact, not nearly."""
    import numpy as np
    from PIL import Image

    import print_grid

    page = print_grid.grid_page(Image.new("RGB", (100, 100), "white"), 3, 2, dpi=50, line=3)
    assert page.size == (150, 100)
    pixels = np.asarray(page).astype(int).sum(axis=2)
    dark = np.flatnonzero(pixels[25] < 300)                 # a row between the horizontal lines
    centres = [int(run.mean()) for run in np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1)]
    assert centres == [0, 50, 100, 149] or centres == [0, 50, 100, 150]


@pytest.mark.parametrize("model", comfy_common.MODELS)
def test_no_links_to_missing_nodes(model):
    """A dangling link kills ComfyUI's job thread (homeai/art/comfy.py: dangling_links)."""
    from homeai.art.comfy import dangling_links

    assert dangling_links(comfy_common.workflow(model, "a fox", 7)) == []
