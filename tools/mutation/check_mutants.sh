#!/usr/bin/env bash
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
#
# Prove the tests guard the behaviour they claim to, by breaking it.
#
# Each mutant is a sed expression applied to one source file. The named tests
# MUST fail with the mutant in place; if they still pass, the test is
# decorative. The source file is always restored, even on Ctrl-C.
#
# Usage: tools/mutation/check_mutants.sh            # run all mutants
#        tools/mutation/check_mutants.sh followup   # names containing "followup"
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

# name | file | sed expression | pytest selector
MUTANTS=(
  "dismissal-off|homeai/daemon.py|s/if is_dismissal(transcript.text/if False and is_dismissal(transcript.text/|tests/test_daemon.py -k dismissal"
  "followup-never-arms|homeai/daemon.py|s/        self._followup_armed.set()/        pass/|tests/test_daemon.py -k followup_window"
  "followup-ignores-chain|homeai/daemon.py|s/if not self._followup_chain.may_open():/if False:/|tests/test_daemon.py -k chain"
  "followup-ignores-interrupt|homeai/daemon.py|s/if not self.cfg.wake.followup or self._last_interrupted:/if not self.cfg.wake.followup:/|tests/test_daemon.py -k interrupted"
  "followup-sends-silence|homeai/daemon.py|s/if source == \"followup\" and not heard:/if False:/|tests/test_daemon.py -k unanswered"
  "lead-in-override-ignored|homeai/wake.py|s/lead_in = self.cfg.lead_in_s if self._lead_in_s is None else self._lead_in_s/lead_in = self.cfg.lead_in_s/|tests/test_wake.py -k lead_in_override"
  "invites-on-first-sentence|homeai/dialogue.py|s/return sentences\[-1\]/return sentences[0]/|tests/test_dialogue.py"
  "budget-ignored|homeai/daemon.py|s/spoken, rest = split_for_budget(spoken, self.cfg.tts.spoken_budget_words)/rest = \"\"/|tests/test_daemon.py -k cut_and_offers"
  "continue-asks-agent|homeai/daemon.py|s/            if is_continue_request(transcript.text):/            if False:/|tests/test_daemon.py -k yes_speaks"
  "decline-ignored|homeai/daemon.py|s/            if is_decline(transcript.text):/            if False:/|tests/test_daemon.py -k declines"
  "stale-rest-kept|homeai/daemon.py|s/            self._held.clear()$/            pass/|tests/test_daemon.py -k drops"
  "split-drops-sentence|homeai/dialogue.py|s/rest = sentences\[len(head):\]/rest = sentences[len(head) + 1:]/|tests/test_dialogue.py -k budget"
  "held-never-expires|homeai/dialogue.py|s/return bool(self.text) and (now - self.at) <= self.ttl_s/return bool(self.text)/|tests/test_dialogue.py -k expires"
  "yes-thanks-declines|homeai/dialogue.py|s/if not words or words\[0\] in _YES_WORDS:/if not words:/|tests/test_dialogue.py -k Continue"
  "worker-dies-on-apology|homeai/daemon.py|s/^                except Exception:  # noqa: BLE001$/                except ZeroDivisionError:/|tests/test_daemon.py -k worker"
  "mic-left-muted|homeai/daemon.py|s/                self.mic.resume()/                pass/|tests/test_daemon.py -k mic"
  "listener-left-running|homeai/daemon.py|s/                listener.stop()/                pass/|tests/test_daemon.py -k listener"
  "no-progress-cue|homeai/daemon.py|s/                self._speak(MSG_WORKING)/                pass/|tests/test_daemon.py -k slow_answer"
  "detector-not-reset|homeai/daemon.py|s/                self._detector.reset()/                pass/|tests/test_daemon.py -k detector_is_reset"
  "research-not-speakable|homeai/research.py|s/will be spoken aloud/will be shown/|tests/test_research.py"
  "mcp-unknown-tool-crashes|homeai/mcp_server.py|s/if handler is None:/if False and handler is None:/|tests/test_mcp_server.py -k unknown_tool"
  "mcp-parse-error-kills-loop|homeai/mcp_server.py|s/response = _error(None, PARSE_ERROR, \"invalid JSON\")/raise/|tests/test_mcp_server.py -k malformed_json"
  "mcp-research-sources-unclamped|homeai/mcp_server.py|s/limit = max(1, min(6, int(limit)))/limit = int(limit)/|tests/test_mcp_server.py -k sources_are"
  "token-required-for-cli|homeai/config.py|s/if self.agent.transport == \"http\" and not self.agent.token:/if not self.agent.token:/|tests/test_stt_config.py -k token"
  "install-vram-boundary|homeai/install/assess.py|s/if vram >= VRAM_FOR_16K_GB and/if vram > VRAM_FOR_16K_GB and/|tests/test_install_assess.py -k boundaries"
  "install-weak-machine-accepted|homeai/install/assess.py|s/    return Assessment(not blockers, tier,/    return Assessment(True, tier,/|tests/test_install_assess.py -k underpowered"
  "install-audio-not-required|homeai/install/assess.py|s/(blockers if require_audio else warnings).append(msg)/warnings.append(msg)/|tests/test_install_assess.py -k all_blockers"
  "install-voice-can-recall|homeai/install/zc_config.py|s/allowed_tools = \[\"calculator\"\]/allowed_tools = [\"calculator\", \"memory_recall\"]/|tests/test_install_zc_config.py -k locked_down"
  "install-redefines-agent|homeai/install/zc_config.py|s/        elif _has_path(data, path):/        elif False:/|tests/test_install_zc_config.py -k rerun"
  "install-overwrites-mcp-server|homeai/install/zc_config.py|s/if server.get(\"command\") != expected_cmd:/if False:/|tests/test_install_zc_config.py -k conflicting"
  "mode-no-rollback|homeai/gpu_mode.py|s/                self.voice()$/                pass/|tests/test_gpu_mode.py -k rolls_back"
  "mode-jarvis-left-on|homeai/gpu_mode.py|s/self.sys.systemctl(\"stop\", VOICE_SERVICE)/pass/|tests/test_gpu_mode.py -k order"
  "mode-voice-before-coder-ready|homeai/gpu_mode.py|s/            self._restart_coder()\$/            self.sys.systemctl(\"restart\", CODER_SERVICE)/|tests/test_gpu_mode.py -k restores_default"
  "mem-no-compact|homeai/memory_privacy.py|s/^        self.be.compact()$/        pass/|tests/test_memory_privacy.py -k purge"
  "mem-ignore-confirm|homeai/memory_privacy.py|s/if confirm is not None and not confirm(n):/if False:/|tests/test_memory_privacy.py -k cancelled"
  "mem-no-verify-clear|homeai/memory_privacy.py|s/        if left:/        if False:/|tests/test_memory_privacy.py -k left_behind"
  "mem-negative-retention|homeai/memory_privacy.py|s/if days < 0:/if days < -99:/|tests/test_memory_privacy.py -k negative"
  "tts-no-voices-dir|homeai/config.py|s/return extra if not direct.exists() and extra.exists() else direct/return direct/|tests/test_tts.py -k VoiceLocation"
  "fragment-ignored|homeai/daemon.py|s/        if is_trailing_fragment(transcript.text):/        if False:/|tests/test_daemon.py -k fragment"
  "fragment-unanchored|homeai/dialogue.py|s/]\\*\$\")/]*\")/|tests/test_dialogue.py -k TrailingFragment"
  "wake-no-flush|homeai/wake.py|s/                for _ in range(FLUSH_FRAMES):/                for _ in range(0):/|tests/test_wake.py -k flushes"
  "transcript-numpy-score|homeai/transcript.py|s/round(float(turn.wake_score), 3)/round(turn.wake_score, 3)/|tests/test_transcript.py -k numpy"
  "wake-numpy-score|homeai/wake.py|s/self.last_score = float(max(scores.values(), default=0.0))/self.last_score = max(scores.values(), default=0.0)/|tests/test_wake.py -k last_score"
  "verify-skipped|homeai/daemon.py|s/            if not mentions_wake_word(transcript.text, self.cfg.wake.model):/            if False:/|tests/test_daemon.py -k wake"
  "verify-no-strip|homeai/daemon.py|s/text=strip_wake_phrase(transcript.text, self.cfg.wake.model))/text=transcript.text)/|tests/test_daemon.py -k verified_wake"
  "verify-no-preroll|homeai/daemon.py|s/            audio = np.concatenate(\[preroll, audio\])/            pass/|tests/test_daemon.py -k verified_wake"
  "verify-loose-match|homeai/wake_verify.py|s/^WAKE_SIMILARITY = 0.75/WAKE_SIMILARITY = 0.5/|tests/test_wake_verify.py"
  "verify-no-prefix|homeai/wake_verify.py|s/        or (len(name) >= 5 and token.startswith(name\[:4\]))/        or False/|tests/test_wake_verify.py"
  "pleasantry-ignores-conversation|homeai/daemon.py|s/            in_conversation = bool(self.memory.recent())/            in_conversation = True/|tests/test_daemon.py -k out_of_nowhere"
  "closing-sent-to-agent|homeai/daemon.py|s/        if is_pleasantry_only(transcript.text):/        if is_pleasantry_only(transcript.text) and not self.memory.recent():/|tests/test_daemon.py -k acknowledge"
  "hallucination-gate-off|homeai/daemon.py|s/            ack = \"\" if is_hallucination_only(transcript.text) or not in_conversation/            ack = \"\" if not in_conversation/|tests/test_daemon.py -k hallucination"
  "reasoning-not-checked|homeai/agent_cli.py|s/            if detect_leaked_reasoning(text):/            if False:/|tests/test_agent_cli.py -k reasoning"
)

filter="${1:-}"
fail=0
for entry in "${MUTANTS[@]}"; do
  IFS='|' read -r name file expr selector <<<"$entry"
  [[ -n "$filter" && "$name" != *"$filter"* ]] && continue
  cp "$file" "$file.mutbak"
  trap 'mv -f "$file.mutbak" "$file"' EXIT INT TERM
  sed -i "$expr" "$file"
  if cmp -s "$file" "$file.mutbak"; then
    echo "STALE    $name  (sed matched nothing; update the mutant)"
    fail=1
  else
    $PY -m pytest $selector -q -p no:warnings -x >/dev/null 2>&1
    rc=$?
    # Exit 5 is "no tests collected": a selector typo would otherwise count
    # as a kill and the mutant would prove nothing.
    if [[ $rc -eq 5 ]]; then
      echo "NOTESTS  $name  (selector '$selector' matched no tests)"
      fail=1
    elif [[ $rc -eq 0 ]]; then
      echo "SURVIVED $name  (tests still pass with the bug in place)"
      fail=1
    else
      echo "killed   $name"
    fi
  fi
  mv -f "$file.mutbak" "$file"
  trap - EXIT INT TERM
done
exit $fail
