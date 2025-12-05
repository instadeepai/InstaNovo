from __future__ import annotations

import os
import shutil
import sys

from rich.console import Console

__version__ = "1.2.0"

# Respect memory limits when running in a container
# The resource module is Unix-specific. We guard this with sys.platform
# so mypy ignores this block when type checking on Windows.
if sys.platform != "win32":
    try:
        import resource

        # Check for cgroup v2 memory file, common in modern container environments
        cgroup_max_mem_limit_path = "/sys/fs/cgroup/memory.max"
        cgroup_high_mem_limit_path = "/sys/fs/cgroup/memory.high"
        if os.path.isfile(cgroup_max_mem_limit_path):
            with open(cgroup_max_mem_limit_path) as limit:
                try:
                    high_limit = open(cgroup_high_mem_limit_path).read()
                    max_limit = open(cgroup_max_mem_limit_path).read()
                    hard_limit = resource.RLIM_INFINITY if max_limit == "max\n" else int(max_limit)
                    soft_limit = hard_limit if high_limit == "max\n" else int(high_limit)

                    # Use RLIMIT_DATA instead of RLIMIT_AS.
                    # RLIMIT_DATA limits the heap size (data segment), which is a better
                    # proxy for actual memory usage and less likely to conflict with
                    # the virtual memory reservation strategies of native libraries
                    # like Polars and PyArrow. RLIMIT_AS is too restrictive and can
                    # cause crashes (core dumps) when these libraries reserve large
                    # virtual address spaces.
                    resource.setrlimit(resource.RLIMIT_DATA, (soft_limit, hard_limit))
                except ValueError:
                    pass
    except ImportError:
        # The 'resource' module is not available on all platforms
        pass

# Get terminal width, default to 175 if not available
terminal_width = shutil.get_terminal_size(fallback=(175, 24)).columns

console = Console(width=terminal_width)

# Global rank variable for distributed training loggers
_rank = None
_rank_set = False


def get_rank() -> int | None:
    """Get the current process rank in distributed training.

    Returns:
        int | None: The process rank if in distributed training, None otherwise
    """
    return _rank


def set_rank(rank: int | None) -> None:
    """Set the current process rank for distributed training.

    Args:
        rank (int | None): The process rank to set, or None for non-distributed training

    Raises:
        RuntimeError: If the rank has already been set
    """
    global _rank, _rank_set
    if _rank_set:
        raise RuntimeError("Rank has already been set. The rank should only be set once during initialization.")
    _rank = rank
    _rank_set = True
