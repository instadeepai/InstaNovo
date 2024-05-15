"""Generate the code reference pages and navigation."""

# This script is used by the mkdocs-gen-files plugin (https://oprypin.github.io/mkdocs-gen-files/)
# for MkDocs (https://www.mkdocs.org/). It creates for each module in the code a stub page
# and it creates a "docs/reference/SUMMARY.md" page which contains a Table of Contents with links to
# all the stub pages. When MkDocs runs, it will populate the stub pages with the documentation
# pulled from the docstrings
from __future__ import annotations

import pathlib

import mkdocs_gen_files

# Folders for which we don't want to create code documentation but which can contain *.py files
IGNORE = ("public", "docs", "tests", "scripts", ".")

nav = mkdocs_gen_files.Nav()

src_folder = "."
module = "instanovo"
for path in sorted(pathlib.Path(src_folder).rglob("*.py")):
    module_path = path.relative_to(src_folder).with_suffix("")
    if (
        not any(str(module_path.parent).startswith(ignore) for ignore in IGNORE)
        and module_path.parts[-1].split("_")[-1] != "test"
    ):
        doc_path = path.relative_to(src_folder, module).with_suffix(".md")
        full_doc_path = pathlib.Path("reference", doc_path)

        parts = tuple(module_path.parts)

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

        mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
