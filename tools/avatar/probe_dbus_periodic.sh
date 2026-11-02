#!/usr/bin/env bash
# List session D-Bus messages that repeat periodically (e.g. something poking
# gnome-shell every 10 s and causing compositor stalls).
#
# Why: the avatar window probe found ~110 ms gnome-shell main-thread stalls
# every 10 s. This shows which senders/members fire during a capture so a
# periodic one can be matched to the stall times.
#
# Usage: tools/avatar/probe_dbus_periodic.sh [seconds, default 25]
set -u
secs=${1:-25}
timeout "$secs" dbus-monitor --session 2>/dev/null \
    | awk '/^(signal|method call|method return|error)/ {
               split($2, t, "="); ts = t[2]
               key = $1
               for (i = 3; i <= NF; i++) if ($i ~ /^(sender|destination|interface|member)=/) key = key " " $i
               n[key]++; last[key] = ts
           }
           END { for (k in n) printf "%5d  %s\n", n[k], k }' \
    | sort -rn | head -40
