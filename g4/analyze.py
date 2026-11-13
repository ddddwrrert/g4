"""Cluster failure events from one execution into signatures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FailureSignature:
    id: str
    event: str
    scope: str
    description: str


# Repeated observations of the same event are one cluster, not several directions.
_CATALOG = (
    FailureSignature(
        "output-not-reopened",
        "skipped_reopen",
        "invariant",
        "The run wrote an output and did not open it again.",
    ),
    FailureSignature(
        "sources-not-reconciled",
        "skipped_reconcile",
        "domain",
        "Figures were not tied back to the source files.",
    ),
)


def cluster_failures(behaviors: Sequence[str]) -> list[FailureSignature]:
    """Known failures keep their catalog ids. Any other skipped, missing, or failed event becomes its own signature."""
    present = set(behaviors)
    found = [signature for signature in _CATALOG if signature.event in present]
    known = {signature.event for signature in _CATALOG}
    for event in sorted(present - known):
        if event.startswith(("skipped_", "missing_", "failed_")):
            found.append(
                FailureSignature(
                    id=event.replace("_", "-"),
                    event=event,
                    scope="domain",
                    description=event.replace("_", " "),
                )
            )
    return found


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def signatures_match(left: str, right: str) -> bool:
    """Exact ids match. Close wording matches when at least half the tokens overlap."""
    if left == right:
        return True
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.5
