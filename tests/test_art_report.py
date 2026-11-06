"""The bake-off report must count a safety card as blanked and keep load time out of warm speed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "report_bakeoff", Path(__file__).resolve().parents[1] / "tools" / "art" / "report_bakeoff.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def card() -> np.ndarray:
    """Mid grey that wobbles, with a band of pale text, like the VAE's card."""
    rng = np.random.default_rng(0)
    pixels = 105 + rng.integers(-3, 4, (256, 256, 1)).repeat(3, axis=2)
    pixels[120:136, 30:220] = 200
    return pixels.astype(np.uint8)


def test_card_is_blank():
    assert report.is_blank(card())


def test_white_paper_is_not_blank():
    assert not report.is_blank(np.full((256, 256, 3), 250, np.uint8))


def test_black_background_is_not_blank():
    assert not report.is_blank(np.zeros((256, 256, 3), np.uint8))


def test_busy_picture_is_not_blank():
    rng = np.random.default_rng(1)
    assert not report.is_blank(rng.integers(0, 256, (256, 256, 3)).astype(np.uint8))


def test_tinted_flat_picture_is_not_blank():
    assert not report.is_blank(np.full((256, 256, 3), (200, 120, 60), np.uint8))


def test_first_image_of_a_run_is_not_warm():
    run = {"a": {"m": [{"seconds": 50.0, "image": "x"}, {"seconds": 3.0, "image": "y"}]},
           "b": {"m": [{"seconds": 4.0, "image": "z"}]}}
    row = report.summarise([run])["m"]
    assert row["warm"] == [3.0, 4.0]
    assert row["images"] == 3


def test_each_run_loads_the_model_again():
    run = {"a": {"m": [{"seconds": 50.0, "image": "x"}, {"seconds": 3.0, "image": "y"}]}}
    assert report.summarise([run, run])["m"]["warm"] == [3.0, 3.0]


def test_errors_and_blanks_are_counted():
    run = {"a": {"m": [{"seconds": 9.0, "image": "bad"}, {"seconds": 9.0, "image": "good"}, {"error": "died"}]}}
    row = report.summarise([run], blank=lambda image: image == "bad")["m"]
    assert row["failed"] == 1
    assert dict(row["blanked"]) == {"a": 1}


def test_table_lists_fastest_first():
    run = {"a": {"slow": [{"seconds": 9.0}, {"seconds": 9.0}], "fast": [{"seconds": 9.0}, {"seconds": 1.0}]}}
    text = report.table(report.summarise([run]))
    assert text.index("fast") < text.index("slow")
