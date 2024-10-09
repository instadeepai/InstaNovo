"""Generate the code reference pages and navigation."""

# This script is used by the mkdocs-gen-files plugin (https://oprypin.github.io/mkdocs-gen-files/)
# for MkDocs (https://www.mkdocs.org/). It creates for each module in the code a stub page
# and it creates a "docs/reference/SUMMARY.md" page which contains a Table of Contents with links to
# all the stub pages. When MkDocs runs, it will populate the stub pages with the documentation
# pulled from the docstrings
from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

# Folders for which we don't want to create code documentation but which can contain *.py files
IGNORE_DIRS = ("build", "docs_public", "docs", "tests", "scripts", "utils", ".venv")


def is_ignored_directory(module_path: Path) -> bool:
    """Check if the module path is within any ignored directory."""
    return any(part in IGNORE_DIRS for part in module_path.parts)


def is_ignored_file(module_path: Path) -> bool:
    """Check if the file is a test file or ignored file."""
    return module_path.parts[-1].endswith("_test") or module_path.parts[-1] in (
        "mlflow_auth",
        "types",
        "constants",
    )


def process_python_files(source_directory: str, module_name: str) -> None:
    """Generate documentation paths for Python files in the source directory."""
    nav = mkdocs_gen_files.Nav()

    for python_file in sorted(Path(source_directory).rglob("*.py")):
        relative_module_path = python_file.relative_to(source_directory).with_suffix("")

        if not is_ignored_directory(relative_module_path) and not is_ignored_file(
            relative_module_path
        ):
            doc_path = python_file.relative_to(
                source_directory, module_name
            ).with_suffix(".md")
            full_doc_path = Path("reference", doc_path)

            parts = tuple(relative_module_path.parts)

            if parts[-1] == "__init__":
                parts = parts[:-1]
                doc_path = doc_path.with_name("index.md")
                full_doc_path = full_doc_path.with_name("index.md")
            elif parts[-1] == "__main__":
                continue

            nav[parts] = doc_path.as_posix()

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                ident = ".".join(parts)
                fd.write(f"::: {ident}")

            mkdocs_gen_files.set_edit_path(full_doc_path, ".." / python_file)

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


process_python_files(source_directory=".", module_name="instanovo")
