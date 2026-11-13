"""Parent weights and patch splitting adapted from HyperAgents.

Source: facebookresearch/HyperAgents commit
59a68f672dfb92c74aeb7e61535d776fb36e172d
utils/gl_utils.py select_parent(method="score_child_prop") and
filter_patch_by_files. The arithmetic matches that function, including the
top-3 midpoint, the slope of 10, and the child penalty exp(-(n/8)**3).
License: CC BY-NC-SA 4.0 (see LICENSE.md).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

# generate_loop.run_harness_polyglot: stagedeval_score >= 0.4 continues.
POLYGLOT_STAGED_THRESHOLD = 0.4


def filter_patch_by_files(patch_str: str, target_files: Sequence[str]) -> str:
    """Drop diff blocks whose header names one of target_files.

    Copied from HyperAgents utils/gl_utils.py. A block is removed when its
    ``diff --git`` line contains both ``a/<target>`` and ``b/<target>``.
    """

    lines = patch_str.splitlines()
    filtered_lines: list[str] = []
    include_block = False
    for line in lines:
        if line.startswith("diff --git"):
            include_block = not any(
                f"a/{target}" in line and f"b/{target}" in line for target in target_files
            )
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + "\n"


def score_child_prop_weights(scores: Sequence[float], child_counts: Sequence[int]) -> list[float]:
    """Normalized weights from select_parent(method="score_child_prop")."""

    if len(scores) != len(child_counts):
        raise ValueError("scores and child_counts must have the same length")
    if not scores:
        return []
    ordered = sorted(scores, reverse=True)[:3]
    mid_point = sum(ordered) / len(ordered)
    shaped = [1 / (1 + math.exp(-10 * (score - mid_point))) for score in scores]
    penalties = [math.exp(-((count / 8) ** 3)) for count in child_counts]
    combined = [shaped_score * penalty for shaped_score, penalty in zip(shaped, penalties)]
    total = sum(combined)
    if total > 0:
        return [item / total for item in combined]
    return [1 / len(combined)] * len(combined)


def choose_parent(
    gen_ids: Sequence[str],
    scores: Sequence[float],
    child_counts: Sequence[int],
    rng: random.Random,
) -> str:
    """One draw from the score_child_prop categorical distribution."""

    weights = score_child_prop_weights(scores, child_counts)
    if not gen_ids or len(gen_ids) != len(weights):
        raise ValueError("gen_ids must align with scores")
    return rng.choices(list(gen_ids), weights=weights, k=1)[0]
