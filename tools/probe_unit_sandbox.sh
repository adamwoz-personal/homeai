#!/usr/bin/env bash
# Is a user unit's declared sandbox actually in force? (SANDBOX_PLAN "same review for homeai-avatar and the memory units")
#
# Why: on this Ubuntu (kernel.apparmor_restrict_unprivileged_userns=1) a *user* unit can't build a mount
# namespace, so ProtectHome/ProtectSystem/ReadWritePaths are silently ignored (probe 2026-11-08). homeai.service
# moved to a system unit for that reason. This checks other user units the same way without touching them: it
# copies the unit's sandbox settings onto a throwaway transient unit that only tries to write one scratch file
# in $HOME (removed again) and compares its mount namespace with the caller's.
#
# Writes ~/homeai-bench/unit-sandbox-<time>.txt and prints it.
#
# Usage: tools/probe_unit_sandbox.sh homeai-avatar homeai-memory-backup homeai-memory-maintain zeroclaw
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 UNIT..." >&2; exit 2; }
out=~/homeai-bench/unit-sandbox-$(date +%Y%m%d-%H%M%S).txt
props=(ProtectSystem ProtectHome ReadWritePaths ReadOnlyPaths PrivateTmp NoNewPrivileges)
{
  echo "kernel.apparmor_restrict_unprivileged_userns=$(sysctl -n kernel.apparmor_restrict_unprivileged_userns)"
  for unit in "$@"; do
    args=()
    declared=""
    for p in "${props[@]}"; do
      v=$(systemctl --user show "$unit" -p "$p" --value)
      [ -n "$v" ] || continue
      declared+=" $p=$v"
      args+=(-p "$p=$v")
    done
    marker="$HOME/.homeai-sandbox-probe-$$"
    result=$(systemd-run --user --quiet --wait --pipe --collect "${args[@]}" \
      /bin/sh -c "if touch '$marker' 2>/dev/null; then echo home=writable; rm -f '$marker'; else echo home=read-only; fi;
                  echo own_mount_ns=\$( [ \"\$(readlink /proc/self/ns/mnt)\" = '$(readlink /proc/self/ns/mnt)' ] && echo no || echo yes)" 2>&1 \
      | tr '\n' ' ')
    echo "$unit: declared:${declared:- none} -> $result"
  done
} | tee "$out"
