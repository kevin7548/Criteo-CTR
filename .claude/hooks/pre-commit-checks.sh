#!/usr/bin/env bash
# Pre-commit quality gate for Claude Code Bash tool calls.
# Runs lint and test before allowing `git commit`.

set -u

INPUT=$(cat)

if [ -z "$INPUT" ]; then
  exit 0
fi

IS_COMMIT=$(
  python3 - "$INPUT" <<'PY'
import json
import re
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

tool_input = payload.get("tool_input") or {}
command = tool_input.get("command") or tool_input.get("cmd") or ""
if isinstance(command, str) and re.search(
    r"(?:^|[\s;&|])git\s+commit(?:\s|$)", command
):
    print("yes")
PY
)

if [ "$IS_COMMIT" != "yes" ]; then
  exit 0
fi

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT" || exit 0

LINT_CMD=${CHECK_LINT_CMD:-"uv run ruff check ."}
TEST_CMD=${CHECK_TEST_CMD:-"uv run pytest -q"}

deny_from_log() {
  local check_name="$1"
  local log_file="$2"
  python3 - "$check_name" "$log_file" <<'PY'
import json
import sys
from pathlib import Path

check_name, log_file = sys.argv[1:]
lines = Path(log_file).read_text(errors="replace").splitlines()
details = "\n".join(lines[-40:])
reason = f"PRE-COMMIT GUARD: {check_name} 검사에 실패해 커밋을 차단했습니다."
if details:
    reason += f"\n\n{details}"

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
}, ensure_ascii=False))
PY
}

run_check() {
  local check_name="$1"
  local check_command="$2"
  local log_file

  log_file=$(mktemp "${TMPDIR:-/tmp}/pre-commit-checks.XXXXXX") || return 1
  if bash -lc "$check_command" >"$log_file" 2>&1; then
    rm -f "$log_file"
    return 0
  fi

  deny_from_log "$check_name" "$log_file"
  rm -f "$log_file"
  return 1
}

run_check "lint" "$LINT_CMD" || exit 0
run_check "test" "$TEST_CMD" || exit 0

exit 0
