from pathlib import Path
from typing import Iterable
from typing import List

import pytest

from dtu_denovo_sequencing.utils.io import load_json
from dtu_denovo_sequencing.utils.io import load_pkl
from dtu_denovo_sequencing.utils.io import load_txt
from dtu_denovo_sequencing.utils.io import load_yml
from dtu_denovo_sequencing.utils.io import save_as_json
from dtu_denovo_sequencing.utils.io import save_as_pkl
from dtu_denovo_sequencing.utils.io import save_as_txt
from dtu_denovo_sequencing.utils.io import save_as_yml


@pytest.fixture(scope="module")
def tmp_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create tmp directory to store all the files linked to the tests."""
    # mypy does not understand the output type
    return tmp_path_factory.mktemp("io_test")  # type: ignore


@pytest.mark.parametrize(
    ("iterable", "expected_loaded_list"),
    [
        (["item1", "item2"], ["item1", "item2"]),
        (("item1", "item2"), ["item1", "item2"]),
        ([1, 2], ["1", "2"]),
        (range(2), ["0", "1"]),
    ],
)
def test_save_load_txt(
    request: pytest.FixtureRequest,
    tmp_directory: Path,
    iterable: Iterable,
    expected_loaded_list: List[str],
) -> None:
    """Ensure the loaded iterable contains the items of the saved iterable (as string)."""
    file_path = tmp_directory / f"{request.node.name}.txt"

    save_as_txt(iterable, file_path)

    loaded_list = load_txt(file_path)

    assert loaded_list == expected_loaded_list


def test_save_load_json(tmp_directory: Path) -> None:
    """Ensure a dictonary is not altered by 'save_as_json' / 'load_json'."""
    file_path = tmp_directory / "test_save_load_json.json"
    dict_ = {
        "key1": list(range(10)),
        "key2": ["test"] * 20,
        "key3": {"key1": list(range(10))},
    }

    save_as_json(dict_, file_path)
    dict_loaded = load_json(file_path)

    assert dict_loaded == dict_


def test_save_load_yml(tmp_directory: Path) -> None:
    """Ensure a dictonary is not altered by 'save_as_yml' / 'load_yml'."""
    file_path = tmp_directory / "test_save_load_yml.yml"
    dict_ = {
        "key1": list(range(10)),
        "key2": ["test"] * 20,
        "key3": {"key1": list(range(10))},
    }

    save_as_yml(dict_, file_path)
    dict_loaded = load_yml(file_path)

    assert dict_loaded == dict_


def test_save_load_pkl(tmp_directory: Path) -> None:
    """Ensure a dictonary is not altered by 'save_as_pkl' / 'load_pkl'."""
    file_path = tmp_directory / "test_save_load_pkl.pkl"
    dict_ = {"key1": range(10), "key2": ["test"] * 20, "key3": {"key1": range(10)}}

    save_as_pkl(dict_, file_path)
    dict_loaded = load_pkl(file_path)

    assert dict_loaded == dict_
