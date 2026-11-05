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


@pytest.mark.parametrize("model", comfy_common.MODELS)
def test_prompt_seed_and_size_reach_the_graph(model):
    graph = comfy_common.workflow(model, "a fox", 7, 640, 480)
    assert graph["pos"]["inputs"]["text"] == "a fox"
    assert graph["sample"]["inputs"]["seed"] == 7
    assert (graph["latent"]["inputs"]["width"], graph["latent"]["inputs"]["height"]) == (640, 480)


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        comfy_common.workflow("dalle", "a fox", 7)


def test_z_image_gets_its_sampling_shift():
    graph = comfy_common.workflow("zimage", "a fox", 7)
    assert graph["sample"]["inputs"]["model"] == ["shift", 0]
