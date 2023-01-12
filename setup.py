from __future__ import annotations

from codecs import open

import setuptools
from setuptools import setup

__version__ = "0.1.0"


def read_requirements(file: str) -> list[str]:
    """Returns content of given requirements file."""
    return [line for line in open(file) if not (line.startswith("#") or line.startswith("--"))]


setup(
    name="dtu_denovo_sequencing",
    version=__version__,
    description="De novo sequencing with advanced AI",
    author="InstaDeep",
    url="https://gitlab.com/instadeep/dtu-denovo-sequencing",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements.txt"),
    extras_require={"dev": read_requirements("./requirements-dev.txt")},
    include_package_data=True,
)
