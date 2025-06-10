from __future__ import annotations

import shutil
import warnings

from rich.console import Console

# Filter out a SyntaxWarning from pubchempy, see:
# https://github.com/mcs07/PubChemPy/pull/53
warnings.filterwarnings(
    "ignore",
    message=r'"is not" with \'int\' literal\. Did you mean "!="\?',
    category=SyntaxWarning,
    module="pubchempy",
)

# Get terminal width, default to 175 if not available
terminal_width = shutil.get_terminal_size(fallback=(175, 24)).columns

console = Console(width=terminal_width)

__version__ = "1.1.3"
