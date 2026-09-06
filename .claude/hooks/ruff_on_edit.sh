#!/usr/bin/env bash
# PostToolUse hook: lint and format-check any edited Python file.
# Reads the tool payload on stdin and extracts the edited path.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
payload="$(cat)"
file=$(printf '%s' "$payload" | /usr/bin/python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)
[[ "$file" == *.py ]] || exit 0
[[ -f "$file" ]] || exit 0
cd "$ROOT" || exit 0
out=$(uv run ruff check "$file" 2>&1; uv run ruff format --check "$file" 2>&1)
status=$?
if printf '%s' "$out" | grep -qE "(error|would be reformatted|Found [0-9]+ error)"; then
  echo "ruff findings in $file:" >&2
  printf '%s\n' "$out" >&2
  exit 2   # exit 2 surfaces the message back to Claude
fi
exit 0
