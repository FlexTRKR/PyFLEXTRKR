from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_requires_minimum_version_310():
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
    upper_bounds = [spec for spec in requirement.specifier if spec.operator in {"<", "<="}]

    assert upper_bounds, "zarr dependency should include an upper bound for Python 3.10 compatibility"
    assert any(
        (spec.operator == "<" and Version(spec.version) <= Version("3.0.0"))
        or (spec.operator == "<=" and Version(spec.version) < Version("3.0.0"))
        for spec in upper_bounds
    )
