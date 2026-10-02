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
