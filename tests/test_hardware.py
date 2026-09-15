# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for hardware detection.

The point of this module is portability, so the tests deliberately exercise
listings from machines other than the development box. Each fixture below is a
realistic ``aplay -l`` shape for a different class of hardware.
"""

from __future__ import annotations

from homeai.hardware import (
    AlsaDevice,
    detect_output_device,
    parse_aplay,
    rank_playback,
)


# The development box: discrete GPU (HDMI on cards 0-1) pushes the real
# analog output to card 2.
DEV_BOX = """\
card 0: HDMI [HDA ATI HDMI], device 3: HDMI 0 [AW3423DWF]
card 0: HDMI [HDA ATI HDMI], device 7: HDMI 1 [HDMI 1]
card 1: Generic [HD-Audio Generic], device 3: HDMI 0 [HDMI 0]
card 2: Generic_1 [HD-Audio Generic], device 0: ALC897 Analog [ALC897 Analog]
card 2: Generic_1 [HD-Audio Generic], device 1: ALC897 Digital [ALC897 Digital]
"""

# A laptop with no discrete GPU: analog lands on card 0.
LAPTOP = """\
card 0: PCH [HDA Intel PCH], device 0: ALC285 Analog [ALC285 Analog]
card 0: PCH [HDA Intel PCH], device 3: HDMI 0 [HDMI 0]
"""

# A purpose-built assistant: USB speaker should beat the built-in jack.
USB_SPEAKER = """\
card 0: PCH [HDA Intel PCH], device 0: ALC285 Analog [ALC285 Analog]
card 1: Speaker [Jabra Speak 410], device 0: USB Audio [USB Audio]
"""

# A headless server whose only output is HDMI: must NOT be auto-selected.
HDMI_ONLY = """\
card 0: HDMI [HDA ATI HDMI], device 3: HDMI 0 [HDMI 0]
card 0: HDMI [HDA ATI HDMI], device 7: HDMI 1 [HDMI 1]
"""


class TestParse:
    def test_parses_card_and_device_numbers(self):
        devices = parse_aplay(DEV_BOX)
        assert len(devices) == 5
        assert devices[0].card == 0 and devices[0].device == 3
        assert devices[3].alsa_id == "plughw:2,0"

    def test_ignores_headers_and_noise(self):
        noisy = "**** List of PLAYBACK Hardware Devices ****\n" + LAPTOP + "\n\n"
        assert len(parse_aplay(noisy)) == 2

    def test_empty_input(self):
        assert parse_aplay("") == []

    def test_description_combines_card_and_device(self):
        dev = parse_aplay(USB_SPEAKER)[1]
        assert "Jabra Speak 410" in dev.description
        assert "USB Audio" in dev.description


class TestSelection:
    def test_dev_box_picks_analog_not_hdmi(self):
        """Regression: this must equal the value that was once hard-coded."""
        assert detect_output_device(DEV_BOX) == "plughw:2,0"

    def test_laptop_picks_card_zero(self):
        assert detect_output_device(LAPTOP) == "plughw:0,0"

    def test_usb_speaker_beats_builtin_analog(self):
        assert detect_output_device(USB_SPEAKER) == "plughw:1,0"

    def test_hdmi_only_refuses_to_guess(self):
        """Silent-speaker failure is worse than deferring to the mixer."""
        assert detect_output_device(HDMI_ONLY) == "default"

    def test_no_devices_falls_back(self):
        assert detect_output_device("") == "default"

    def test_digital_outranked_by_analog(self):
        ranked = rank_playback(parse_aplay(DEV_BOX))
        assert "Analog" in ranked[0].device_name

    def test_ranking_is_deterministic(self):
        listing = parse_aplay(DEV_BOX)
        assert rank_playback(listing) == rank_playback(list(reversed(listing)))

    def test_unknown_device_scores_between_hdmi_and_analog(self):
        """An unrecognised device is usable, but never preferred over analog."""
        dev = AlsaDevice(0, 0, "Some Codec", "Mystery Output")
        hdmi = AlsaDevice(0, 3, "Some Codec", "HDMI 0")
        analog = AlsaDevice(0, 1, "Some Codec", "Analog Out")
        assert hdmi.score() < dev.score() < analog.score()


class TestPlughwContract:
    def test_uses_plughw_not_hw(self):
        """'hw' fails when Piper's mono 22.05 kHz meets a stereo-only codec."""
        assert AlsaDevice(2, 0, "c", "d").alsa_id.startswith("plughw:")


class TestConfigIntegration:
    def test_env_override_wins_over_detection(self, monkeypatch):
        monkeypatch.setenv("HOMEAI_OUTPUT_DEVICE", "plughw:9,9")
        from homeai.config import AudioConfig

        assert AudioConfig().output_device == "plughw:9,9"

    def test_detection_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("HOMEAI_OUTPUT_DEVICE", raising=False)
        from homeai.config import AudioConfig

        value = AudioConfig().output_device
        assert value == "default" or value.startswith("plughw:")
