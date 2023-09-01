from __future__ import annotations

from codecs import open

import setuptools
from setuptools import setup
from pathlib import Path

__version__ = "0.1.2"


def read_requirements(file: str) -> list[str]:
    """Returns content of given requirements file."""
    return [line for line in open(file) if not (line.startswith("#") or line.startswith("--"))]


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="instanovo",
    version=__version__,
    description="De novo sequencing with InstaNovo",
    author="InstaDeep",
    url="https://githun.com/instadeepai/InstaNovo",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements.txt"),
    # extras_require={"dev": read_requirements("./requirements-dev.txt")},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
