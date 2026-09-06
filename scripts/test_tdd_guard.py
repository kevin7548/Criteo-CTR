import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).parents[1] / ".codex" / "hooks" / "tdd-guard.sh"


def run_guard(tmp_path: Path, *, file_path: str | None = None, patch: str | None = None):
    tool_input = {}
    if file_path is not None:
        tool_input["file_path"] = file_path
    if patch is not None:
        tool_input["command"] = patch

    return subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps({"tool_name": "apply_patch", "tool_input": tool_input}),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )


def test_blocks_python_implementation_without_test(tmp_path):
    (tmp_path / "src").mkdir()

    result = run_guard(tmp_path, file_path="src/widget.py")

    assert result.returncode == 0
    output = json.loads(result.stdout)
    decision = output["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert "test_widget.py" in decision["permissionDecisionReason"]


def test_allows_python_implementation_with_project_test(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").touch()

    result = run_guard(tmp_path, file_path="src/widget.py")

    assert result.returncode == 0
    assert result.stdout == ""


def test_allows_test_file_creation(tmp_path):
    result = run_guard(tmp_path, file_path="tests/test_widget.py")

    assert result.returncode == 0
    assert result.stdout == ""


def test_reads_paths_from_codex_apply_patch_input(tmp_path):
    patch = "*** Begin Patch\n*** Add File: src/widget.ts\n+export const widget = 1;\n*** End Patch"

    result = run_guard(tmp_path, patch=patch)

    output = json.loads(result.stdout)
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "widget.test.ts" in output["hookSpecificOutput"]["permissionDecisionReason"]
