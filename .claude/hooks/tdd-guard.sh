#!/usr/bin/env bash
# TDD Guard Hook — Claude Code PreToolUse[Edit|Write|MultiEdit]
# Blocks edits to Python modules under src/ when no matching test_<module>.py
# exists anywhere under tests/.
#
# Adapted from:
# https://github.com/jha0313/codex-live-demo/blob/feat-0-mvp/.codex/hooks/tdd-guard.sh

set -u

INPUT=$(cat)

if [ -z "$INPUT" ]; then
  exit 0
fi

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

deny() {
  local reason="$1"
  python3 - "$reason" <<'PY'
import json
import sys

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": sys.argv[1],
    }
}, ensure_ascii=False))
PY
}

# Prints the edited path relative to the project root, or nothing when the
# path lies outside the project.
REL_PATH=$(
  python3 - "$INPUT" "$PROJECT_ROOT" <<'PY'
import json
import os
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

file_path = (payload.get("tool_input") or {}).get("file_path")
if not isinstance(file_path, str) or not file_path:
    sys.exit(0)

root = os.path.realpath(sys.argv[2])
if not os.path.isabs(file_path):
    file_path = os.path.join(root, file_path)
rel = os.path.relpath(os.path.realpath(file_path), root)
if not rel.startswith(".."):
    print(rel)
PY
)

case "$REL_PATH" in
  src/*.py) ;;
  *) exit 0 ;;
esac

BASE_NAME=$(basename "$REL_PATH" .py)

case "$BASE_NAME" in
  __init__|conftest|test_*|*_test) exit 0 ;;
esac

if [ -d "${PROJECT_ROOT}/tests" ] && \
   [ -n "$(find "${PROJECT_ROOT}/tests" -name "test_${BASE_NAME}.py" -print -quit)" ]; then
  exit 0
fi

deny "TDD GUARD: '${BASE_NAME}'에 대한 테스트 파일이 존재하지 않습니다. 구현 코드를 작성하기 전에 tests/ 아래에 test_${BASE_NAME}.py를 먼저 작성하세요."
exit 0
