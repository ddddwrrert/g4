"""Put the cloned Harness² tree on the import path without copying it."""

from __future__ import annotations

import sys
from pathlib import Path

HARNESS2_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "harness2"
HARNESS2_COMMIT = "6868817518876de8fd170c6a8354daca273b6191"


def ensure_harness2() -> Path:
    package = HARNESS2_ROOT / "harness2" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(
            "Harness² source is missing at third_party/harness2. "
            "Clone https://github.com/google-research/harness2 and check out the core package."
        )
    entry = str(HARNESS2_ROOT)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return HARNESS2_ROOT
