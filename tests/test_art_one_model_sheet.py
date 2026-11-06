"""The review page must hold every seed of every named scenario."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "one_model_sheet", Path(__file__).resolve().parents[1] / "tools" / "art" / "one_model_sheet.py")
sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheet)


def test_page_stacks_scenarios_and_lays_seeds_side_by_side(tmp_path, monkeypatch):
    for name, colour in (("a", "red"), ("b", "blue")):
        for seed in range(3):
            Image.new("RGB", (100, 100), colour).save(tmp_path / f"{name}-seed{seed}.png")
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), "--scenario", "a", "--scenario", "b"])
    assert sheet.main() == 0
    page = Image.open(tmp_path / "all-review.png")
    assert page.size == (sheet.CELL * 3, sheet.CELL * 2)
    assert page.getpixel((sheet.CELL * 3 - 1, 0)) == (254, 0, 0) or page.getpixel((sheet.CELL * 3 - 1, 0))[0] > 200
    assert page.getpixel((0, sheet.CELL * 2 - 1))[2] > 200
