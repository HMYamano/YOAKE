"""
Repository metadata contracts.

These guard the things a reader trusts without running them: the install
commands in the README, the extras in pyproject.toml, the entry points, and the
CI matrix. Each of them was wrong at some point in a way no code test could
catch — a README that promises ``pip install -e ".[all]"`` installs everything
while ``all`` is missing dearpygui costs somebody an afternoon.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip(
            "tomli", reason="needs tomllib (3.11+) or tomli to read pyproject.toml"
        )
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _workflow() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Dev environment / CI
# --------------------------------------------------------------------------- #

def test_requirements_dev_exists_and_mentions_pytest():
    content = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in content


def test_requirements_dev_installs_the_extras_the_suite_needs():
    """The suite imports torch and dearpygui; the base install has neither."""
    content = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "[ml,gui]" in content or "[gui,ml]" in content, (
        "requirements-dev.txt must pull the ml and gui extras, or CI cannot "
        "import torch/dearpygui"
    )


def test_github_actions_runs_pytest():
    assert "pytest -q" in _workflow()


def test_ci_tests_windows_as_well_as_linux():
    """Windows is the primary environment for the annotator and the sidecar."""
    wf = _workflow()
    assert "windows-latest" in wf, "CI does not test Windows"
    assert "ubuntu-latest" in wf


def test_ci_tests_more_than_one_python_version():
    wf = _workflow()
    assert '"3.10"' in wf
    assert '"3.11"' in wf


def test_ci_byte_compiles_the_tree():
    assert "compileall" in _workflow()


def test_ci_runs_ruff_and_mypy():
    wf = _workflow()
    assert "ruff check" in wf
    assert "mypy" in wf


def test_mypy_clean_list_only_names_existing_files():
    listing = REPO_ROOT / ".mypy-clean.txt"
    assert listing.exists(), "CI reads .mypy-clean.txt"
    missing = [
        line.strip()
        for line in listing.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
        and not (REPO_ROOT / line.strip()).exists()
    ]
    assert missing == [], f"listed for mypy but gone: {missing}"


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #

def test_all_extra_covers_every_installable_extra():
    """
    ``all`` must actually mean all — minus the ones that cannot install
    unattended. It previously omitted dearpygui, so ``pip install -e ".[all]"``
    followed by ``yoake gui`` failed on a fresh machine.
    """
    extras = _pyproject()["project"]["optional-dependencies"]
    all_names = {req.split(">")[0].split("=")[0].split("[")[0].strip().lower()
                 for req in extras["all"]}

    # sam2 is excluded on purpose (see the comment in pyproject.toml), and dev
    # is tooling rather than a runtime feature.
    for group in ("gui", "ml", "vis", "wandb", "concept"):
        for req in extras[group]:
            name = req.split(">")[0].split("=")[0].split("[")[0].strip().lower()
            assert name in all_names, f"'{name}' is in [{group}] but not in [all]"


def test_all_extra_excludes_sam2():
    """sam2 cannot be installed as a plain PyPI dependency; keep it opt-in."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert not any("sam2" in req for req in extras["all"])
    assert any("sam2" in req for req in extras["sam2"])


def test_sam2_extra_carries_huggingface_hub():
    """``pip install sam2`` alone cannot reach a checkpoint without it."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any("huggingface_hub" in req for req in extras["sam2"])


def test_base_install_is_torch_free():
    """
    The GUI is torch-free by design, and extras cannot subtract a base
    dependency — so torch must not be in the base list or ``.[gui]`` drags the
    whole ML stack (and CUDA wheels) onto a labelling machine.
    """
    base = _pyproject()["project"]["dependencies"]
    names = {req.split(">")[0].split("=")[0].strip().lower() for req in base}
    assert "torch" not in names
    assert "torchvision" not in names


def test_gui_extra_is_torch_free():
    extras = _pyproject()["project"]["optional-dependencies"]
    for req in extras["gui"]:
        assert "torch" not in req.lower()


def test_ml_extra_has_the_training_stack():
    extras = _pyproject()["project"]["optional-dependencies"]
    joined = " ".join(extras["ml"]).lower()
    for pkg in ("torch", "torchvision", "scipy"):
        assert pkg in joined


def test_readme_documents_the_extras_it_tells_people_to_install():
    readme = _readme()
    extras = _pyproject()["project"]["optional-dependencies"]
    for group in ("gui", "ml", "all"):
        assert f'".[{group}]"' in readme, f"README never mentions the [{group}] extra"
        assert group in extras


def test_readme_warns_that_base_install_no_longer_brings_torch():
    """A silent change here would look like a broken install to existing users."""
    readme = _readme()
    assert "no longer" in readme and "PyTorch" in readme


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def test_gui_module_is_runnable_with_dash_m():
    """README documents ``python -m htrtdetr.gui``; that needs a __main__."""
    assert (REPO_ROOT / "src" / "htrtdetr" / "gui" / "__main__.py").exists()


def test_every_documented_gui_launcher_resolves():
    import importlib

    scripts = _pyproject()["project"]["scripts"]
    assert scripts["gui"] == "htrtdetr.gui:main"
    assert importlib.import_module("htrtdetr.gui").main is not None
    assert importlib.util.find_spec("htrtdetr.gui.__main__") is not None


def test_console_scripts_all_point_at_something_importable():
    import importlib

    for name, target in _pyproject()["project"]["scripts"].items():
        module_path, _, attr = target.partition(":")
        module = importlib.import_module(module_path)
        assert hasattr(module, attr), f"entry point {name} -> {target} is dead"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def test_data_public_exports_exist():
    import htrtdetr.data as data

    missing = [name for name in data.__all__ if not hasattr(data, name)]
    assert missing == []
