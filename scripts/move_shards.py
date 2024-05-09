from __future__ import annotations

import os
import sys


def rename_files(directory: str, target: str, counter: int) -> None:
    """Rename training files to shards."""
    file_pattern = "train.ipc"
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(file_pattern):
            old_path = os.path.join(directory, filename)
            new_filename = f"train_shard_{counter}.ipc"
            new_path = os.path.join(target, new_filename)
            os.rename(old_path, new_path)
            counter += 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rename_files.py <source> <target> <offset>")
        sys.exit(1)

    rename_files(sys.argv[1], sys.argv[2], int(sys.argv[3]))
