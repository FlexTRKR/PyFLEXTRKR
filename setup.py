import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pyflextrkr",
    version="1.0.0",
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
    python_requires='>=3.6',
)
