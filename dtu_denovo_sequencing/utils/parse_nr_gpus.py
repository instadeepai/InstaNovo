from __future__ import annotations

import yaml
from yaml.loader import SafeLoader


def parse_nr_gpus() -> int:
    """Returns the number of GPUs specified in the manifest.yaml file."""
    with open("manifest.yaml") as f:
        data = yaml.load(f, Loader=SafeLoader)
        nr_gpus = data["spec"]["types"]["Worker"]["gpus"]
    return int(nr_gpus)


if __name__ == "__main__":
    print(parse_nr_gpus())
