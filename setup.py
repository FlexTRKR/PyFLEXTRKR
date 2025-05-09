import setuptools
import os
import re

# Read version from _version.py
with open(os.path.join('pyflextrkr', '_version.py'), 'r') as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pyflextrkr",
    version=version,
    author="Zhe Feng, Hannah Barnes, Joseph Hardin",
    author_email="zhe.feng@pnnl.gov",
    description="A Python package for atmospheric feature tracking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FlexTRKR/PyFLEXTRKR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ],
    install_requires=required,
    python_requires='>=3.10',
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/FlexTRKR/PyFLEXTRKR/issues",
        "Documentation": "https://github.com/FlexTRKR/PyFLEXTRKR",
        "Source Code": "https://github.com/FlexTRKR/PyFLEXTRKR",
    },
)
