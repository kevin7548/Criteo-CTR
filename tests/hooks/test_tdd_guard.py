import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).parents[2] / ".claude" / "hooks" / "tdd-guard.sh"


def run_guard(tmp_path: Path, file_path: str, tool_name: str = "Write"):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps({"tool_name": tool_name, "tool_input": {"file_path": file_path}}),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )


def test_blocks_src_implementation_without_test(tmp_path):
    result = run_guard(tmp_path, "src/ctr/metrics.py")

    assert result.returncode == 0
    decision = json.loads(result.stdout)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert "test_metrics.py" in decision["permissionDecisionReason"]


def test_blocks_absolute_src_path_without_test(tmp_path):
    result = run_guard(tmp_path, str(tmp_path / "src" / "ctr" / "metrics.py"), "Edit")

    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_allows_src_implementation_with_nested_test(tmp_path):
    (tmp_path / "tests" / "data").mkdir(parents=True)
    (tmp_path / "tests" / "data" / "test_split.py").touch()

    result = run_guard(tmp_path, "src/ctr/data/split.py")

    assert result.returncode == 0
    assert result.stdout == ""


def test_allows_test_file_creation(tmp_path):
    result = run_guard(tmp_path, "tests/test_metrics.py")

    assert result.stdout == ""


def test_allows_init_module(tmp_path):
    result = run_guard(tmp_path, "src/ctr/__init__.py")

    assert result.stdout == ""


def test_allows_python_outside_src(tmp_path):
    assert run_guard(tmp_path, "scripts/make_report.py").stdout == ""
    assert run_guard(tmp_path, "/tmp/scratch/explore.py").stdout == ""


def test_does_not_skip_module_whose_name_contains_test(tmp_path):
    result = run_guard(tmp_path, "src/ctr/latest_block.py")

    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
