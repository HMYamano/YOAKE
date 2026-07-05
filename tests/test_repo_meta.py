from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_dev_exists_and_mentions_pytest():
    content = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in content


def test_github_actions_runs_pytest():
    content = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "pytest -q" in content


def test_data_public_exports_exist():
    import htrtdetr.data as data

    missing = [name for name in data.__all__ if not hasattr(data, name)]
    assert missing == []
