"""Define functions for common I/O operations."""
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Union

import yaml

Openable = Union[str, os.PathLike]


def save_as_txt(iterable: Iterable, file_path: Openable, force: bool = False) -> None:
    """Save an iterable to a .txt file.

    Each item of the iterable is converted to str and written to a new line of the file.

    Args:
        iterable: iterable to save on disk.
        file_path: path to the file.
        force: indicates if the file is erased in case it already exists.

    Raises:
        FileExistsError: if the 'file_path' exists and 'force' is set to False
    """

    def save_fn(iterable: Iterable, f: io.TextIOBase) -> None:
        for item in iterable:
            f.write(f"{item}\n")

    _save(obj=iterable, file_path=file_path, force=force, save_fn=save_fn)


def load_txt(file_path: Openable) -> List[str]:
    """Load a .txt file and store it in a list.

    Each line corresponds to an item in the returned list.

    Args:
        file_path: path to the file to load.

    Raises:
        FileNotFoundError: if the 'file_path' does not exist.
    """

    return _load(  # type: ignore
        file_path=file_path, load_fn=lambda f: [item.strip() for item in f]
    )


def save_as_json(
    dict_: dict, file_path: Openable, force: bool = False, **save_kwargs: Any
) -> None:
    """Save a dictionary to a .json file thanks to 'json' package.

    By default the 'indent' in 'save_kwargs' is set to 4 for a better display.

    Args:
        dict_: dictionary to save on disk.
        file_path: path to the file.
        force: indicates if the file is erased in case it already exists.
        save_kwargs: additional parameters to forward to 'json.dump'

    Raises:
        FileExistsError: if the 'file_path' exists and 'force' is set to False.
    """
    save_kwargs["indent"] = save_kwargs.get("indent", 4)

    _save(obj=dict_, file_path=file_path, save_fn=json.dump, force=force, **save_kwargs)


def load_json(file_path: Openable, **load_kwargs: Any) -> dict:
    """Load a .json file and store it in a dictionary.

    Args:
        file_path: path to the file to load.
        load_kwargs: additional parameters to forward to 'json.load'

    Raises:
        FileNotFoundError: if the 'file_path' does not exist.
    """
    return _load(file_path=file_path, load_fn=json.load, **load_kwargs)  # type: ignore


def save_as_yml(
    dict_: dict, file_path: Openable, force: bool = False, **save_kwargs: Any
) -> None:
    """Save a dictionary to a .yml file thanks to 'yaml' package.

    By default 'default_flow_style' and 'sort_keys' are set to False in 'save_kwargs'.

    Args:
        dict_: dictionary to save on disk.
        file_path: path to the file.
        force: indicates if the file is erased in case it already exists.
        save_kwargs: additional parameters to forward to 'yaml.dump'.

    Raises:
        FileExistsError: if the 'file_path' exists and 'force' is set to False.
    """
    save_kwargs["default_flow_style"] = save_kwargs.get("default_flow_style", False)
    save_kwargs["sort_keys"] = save_kwargs.get("sort_keys", False)

    _save(obj=dict_, file_path=file_path, save_fn=yaml.dump, force=force, **save_kwargs)


def load_yml(file_path: Openable, **load_kwargs: Any) -> dict:
    """Load a .yml file and store it in a dictionary.

    Args:
        file_path: path to the file to load.
        load_kwargs: additional parameters to forward to 'yaml.full_load'.

    Raises:
        FileNotFoundError: if the 'file_path' does not exist.
    """
    return _load(  # type: ignore
        file_path=file_path, load_fn=yaml.full_load, **load_kwargs
    )


def save_as_pkl(
    obj: Any, file_path: Openable, force: bool = False, **save_kwargs: Any
) -> None:
    """Save an object into a .pkl file.

    Args:
        obj: any python object.
        file_path: path to the file.
        force: indicates if the file is erased in case it already exists.
        save_kwargs: additional parameters to forward to 'pickle.dump'.

    Raises:
        FileExistsError: if the 'file_path' exists and 'force' is set to False.
    """
    _save(
        obj=obj,
        file_path=file_path,
        save_fn=pickle.dump,
        force=force,
        file_mode="wb",
        **save_kwargs,
    )


def load_pkl(file_path: Openable, **load_kwargs: Any) -> Any:
    """Load a .pkl file and store it in a Python object.

    Args:
        file_path: path to the file to load.
        load_kwargs: additional parameters to forward to 'pickle.load'

    Raises:
        FileNotFoundError: if the 'file_path' does not exist.
    """
    return _load(
        file_path=file_path, load_fn=pickle.load, file_mode="rb", **load_kwargs
    )


def _save(
    obj: Any,
    file_path: Openable,
    save_fn: Callable,
    force: bool = False,
    file_mode: str = "w",
    **save_kwargs: Any,
) -> None:
    """Main function to save a python object thanks to the provided save_fn.

    Args:
        obj: any Python object
        file_path: path to the file.
        save_fn: function used to save the object.
        force: indicates if the file is erased in case it already exists.
        file_mode: mode used to open the file.
        save_kwargs: additional parameters to forward to 'save_fn'.

    Raises:
        FileExistsError: if the 'file_path' exists and 'force' is set to False.
    """
    if not force and Path(file_path).exists():
        raise FileExistsError(
            f"The file '{file_path}' already exists. Use force=True to overwrite it."
        )

    # Make sure all the subdirectories exist otherwise Python will raise a FileNotFoundError
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, file_mode) as f:
        save_fn(obj, f, **save_kwargs)


def _load(
    file_path: Openable, load_fn: Callable, file_mode: str = "r", **load_kwargs: Any
) -> Any:
    """Main function to load a python object thanks to the provided load_fn.

    Args:
        file_path: path to the file to load.
        load_fn: function used to load the object.
        load_kwargs: additional parameters to forward to 'load_fn'.
        file_mode: mode used to open the file.

    Raises:
        FileNotFoundError: if the 'file_path' does not exist.
    """
    with open(file_path, file_mode) as f:
        obj = load_fn(f, **load_kwargs)

    return obj
