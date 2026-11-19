"""The LoRA bench compares like with like: 4B plain, 4B + LoRA plain, 9B with the glossary; and the training
set keeps only pictures whose every answer was read and right."""
from __future__ import annotations

import pytest

from homeai.art import wording as W
from homeai.art import workflows
from tools.art import bench_lora as B
from tools.art import collect_lora_set as L


def test_4b_plain_is_klein4_with_the_plain_words():
    g = B.graph("4b", "a chimera", 3)
    assert g["unet"]["inputs"]["unet_name"] == B.KLEIN4_UNET and g["clip"]["inputs"]["clip_name"] == B.KLEIN4_CLIP
    assert g["pos"]["inputs"]["text"] == "a chimera" and "lora" not in g
    assert g["guider"]["inputs"]["model"] == ["unet", 0]
    assert workflows.klein9("x", 1)["unet"]["inputs"]["unet_name"] == workflows.KLEIN9_UNET    # not mutated


def test_4b_lora_passes_the_model_through_the_lora():
    g = B.graph("4b+lora", "a chimera", 3, "homeai-c.safetensors", 0.8)
    assert g["lora"]["inputs"] == {"model": ["unet", 0], "lora_name": "homeai-c.safetensors", "strength_model": 0.8}
    assert g["guider"]["inputs"]["model"] == ["lora", 0]
    assert g["pos"]["inputs"]["text"] == "a chimera"
    with pytest.raises(ValueError):
        B.graph("4b+lora", "a chimera", 3)


def test_9b_gets_the_glossary_description():
    g = B.graph("9b+glossary", "a chimera", 3)
    assert g["unet"]["inputs"]["unet_name"] == workflows.KLEIN9_UNET
    assert g["pos"]["inputs"]["text"] == W.expand("a chimera")[0] != "a chimera"


def test_the_set_is_strict():
    goat, horns = W.expand("a chimera")[1][0], W.expand("a chimera")[1][2]
    assert L.right([(goat, "Yes."), (horns, "No")])
    assert not L.right([(goat, "Yes."), (horns, "Yes")])
    assert not L.right([(goat, "I can't tell"), (horns, "No")])       # unreadable drops it
    blades = W.expand("a girl wielding a katana")[1][0]
    assert L.right([(blades, "1")]) and not L.right([(blades, "2")])


def test_the_captions_are_plain_names():
    for plain in L.SCENES.values():
        assert "chimera" in plain and "goat" not in plain and W.expand(plain)[0] != plain
