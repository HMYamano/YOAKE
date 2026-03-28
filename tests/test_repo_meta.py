from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_dev_exists_and_mentions_pytest():
    content = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in content


def test_github_actions_runs_pytest():
    content = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "pytest -q" in content
