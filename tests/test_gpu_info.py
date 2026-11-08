"""homeai/gpu_info.py: AMD VRAM from sysfs for the mode switcher, the installer and the benchmarks."""
from __future__ import annotations

from homeai import gpu_info as GI


def card(root, name, total=None, used=None):
    device = root / name / "device"
    device.mkdir(parents=True)
    if total is not None:
        (device / "mem_info_vram_total").write_text(f"{total}\n")
    if used is not None:
        (device / "mem_info_vram_used").write_text(f"{used}\n")


def test_reads_every_card_in_bytes_in_name_order(tmp_path):
    card(tmp_path, "card1", 20 * 2**30, 5 * 2**30)
    card(tmp_path, "card0", 512 * 2**20, 100)
    card(tmp_path, "card1-DP-1")                     # a connector, not a card
    assert GI.amd_cards(tmp_path) == [GI.Card("card0", 512 * 2**20, 100), GI.Card("card1", 20 * 2**30, 5 * 2**30)]


def test_unreadable_cards_are_skipped_and_missing_used_is_zero(tmp_path):
    card(tmp_path, "card0", "garbage", 1)
    card(tmp_path, "card2", 1000)
    assert GI.amd_cards(tmp_path) == [GI.Card("card2", 1000, 0)]


def test_biggest(tmp_path):
    assert GI.biggest(tmp_path) is None
    card(tmp_path, "card0", 512, 1)
    card(tmp_path, "card1", 2048, 7)
    assert GI.biggest(tmp_path) == GI.Card("card1", 2048, 7)


def test_the_mode_switcher_reports_the_biggest_card_in_gib(tmp_path, monkeypatch):
    from homeai import gpu_mode as G
    card(tmp_path, "card0", 512 * 2**20, 0)
    card(tmp_path, "card1", 20 * 2**30, 5 * 2**30)
    monkeypatch.setattr(GI, "DRM", tmp_path)
    monkeypatch.setattr(GI.biggest, "__defaults__", (tmp_path,))
    assert G.System().vram_gb() == (20.0, 5.0)
