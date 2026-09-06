#!/usr/bin/env bash
# Paper style gate. Fails if banned punctuation or filler appears in paper/**/*.tex.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 0
# macOS ships bash 3.2, which has no globstar, so recurse with find instead.
IFS=$'\n' read -r -d '' -a files < <(find paper -name '*.tex' -type f 2>/dev/null && printf '\0')
[[ ${#files[@]} -eq 0 ]] && exit 0

fail=0
report() { echo "PROSE CHECK FAILED: $1" >&2; fail=1; }

if grep -n -- "—" "${files[@]}" 2>/dev/null; then report "em dash (—) found; use commas, colons or separate sentences"; fi
if grep -n -- "–" "${files[@]}" 2>/dev/null; then report "en dash (–) found; use commas, colons or separate sentences"; fi
for word in delve leverage "it is worth noting" "in the realm of"; do
  if grep -n -i -- "$word" "${files[@]}" 2>/dev/null; then report "banned filler: '$word'"; fi
done
# Sentences starting with "And"
if grep -n -E '(^|[.!?]["'"'"']?[[:space:]]+)And[[:space:]]' "${files[@]}" 2>/dev/null; then
  report "sentence starting with 'And'"
fi
# Bullet lists in the paper body (appendix feature dictionary may use tables instead)
if grep -n -E '\\begin\{(itemize|enumerate)\}' "${files[@]}" 2>/dev/null; then
  report "bullet/enumerate list in paper body"
fi
exit $fail
