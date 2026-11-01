#!/usr/bin/env bash
# Have the ZeroClaw builder agent use the memory MCP tools end to end (Phase 3
# functionality check), then verify in the database what it actually did.
#
#   tools/memory/builder_memory_check.sh
#
# The agent is asked to add, find, supersede and delete a marker fact in a
# throwaway thread. Its words are not trusted: the check reads the audit log
# (actor mcp:builder) and confirms nothing with the marker is left.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

marker="builderprobe$$"
thread="builder probe $$"
since=$(date -u +%FT%TZ)
# A fresh session per run: reused history poisons tool calling (ROUTING_GUIDE).
state=$(mktemp -t builder-memcheck.XXXXXX.json)
rm -f "$state"
trap 'rm -f "$state"' EXIT
prompt="Use your memory tools (memory_* via the memory MCP server). Steps:
1. memory_health and report the newest backup name.
2. memory_add the fact \"Adam's ${marker} test color is teal\" with topics [\"${thread}\"] and thread \"${thread}\".
3. memory_search for \"${marker} test color\" and give the id you find.
4. memory_supersede that id with \"Adam's ${marker} test color is cyan\".
5. memory_topics with prefix \"${thread}\".
6. memory_delete with thread \"${thread}\" and reason \"dead_end\".
Report each tool's result in one line. Do not use any other tools."

timeout 900 zeroclaw agent --agent builder --session-state-file "$state" -m "$prompt" 2>&1 \
  | tail -40 || true

echo "---- audit (actor mcp:builder since ${since}) ----"
psql -X -q -At -d homeai -c "SET app.scope = 'admin';" \
  -c "SELECT op || ' ' || coalesce(detail::text, '') FROM memory.events
      WHERE actor = 'mcp:builder' AND at >= '${since}' ORDER BY id;"
left=$(psql -X -q -At -d homeai -c "SET app.scope = 'admin';" \
  -c "SELECT count(*) FROM memory.facts WHERE summary LIKE '%${marker}%';")
ops=$(psql -X -q -At -d homeai -c "SET app.scope = 'admin';" \
  -c "SELECT string_agg(DISTINCT op, ',' ORDER BY op) FROM memory.events
      WHERE actor = 'mcp:builder' AND at >= '${since}';")
echo "facts left with marker: ${left}; ops: ${ops}"
for op in fact.add fact.correct thread.forget; do
  [[ ",${ops}," == *",${op},"* ]] || { echo "FAIL: builder never did ${op}"; exit 1; }
done
[[ "${left}" == "0" ]] || { echo "FAIL: marker facts left behind"; exit 1; }
echo "builder memory check: ok"
