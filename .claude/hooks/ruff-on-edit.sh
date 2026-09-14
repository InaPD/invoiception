#!/usr/bin/env bash
# PostToolUse hook: format and lint-fix Python files after Claude edits them.
# Never blocks: always exits 0, even when ruff is missing or the file is unparseable.
set -uo pipefail

payload="$(cat)"

file_path="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print(data.get("tool_input", {}).get("file_path", "") or "")
' 2>/dev/null)"

# Only Python files, and only ones that still exist on disk.
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file_path" ] || exit 0

# Prefer a ruff on PATH; fall back to uvx so the hook works on a clean machine.
if command -v ruff >/dev/null 2>&1; then
  ruff_cmd=(ruff)
elif command -v uvx >/dev/null 2>&1; then
  ruff_cmd=(uvx ruff)
else
  exit 0
fi

"${ruff_cmd[@]}" format --quiet "$file_path" 2>/dev/null
"${ruff_cmd[@]}" check --quiet --fix "$file_path" 2>/dev/null

# Report anything ruff could not fix itself, as feedback rather than a failure.
remaining="$("${ruff_cmd[@]}" check --quiet "$file_path" 2>/dev/null)"
if [ -n "$remaining" ]; then
  echo "[ruff] unresolved lint in $file_path:" >&2
  printf '%s\n' "$remaining" >&2
fi

exit 0
