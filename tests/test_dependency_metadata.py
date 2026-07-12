from pathlib import Path
import re

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_requires_remains_310_compatible():
    setup_text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", setup_text)
    assert match is not None
    assert match.group(1) == ">=3.10"


def test_requirements_keep_zarr_python310_compatible():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    zarr_line = None

    for line in requirements:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.startswith("zarr"):
            zarr_line = stripped
            break

    assert zarr_line is not None, "requirements.txt should declare a zarr dependency"
    requirement = Requirement(zarr_line)
    assert requirement.name == "zarr"
    assert requirement.specifier.contains("2.18.7")
    assert not requirement.specifier.contains("3.0.0")
