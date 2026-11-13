"""Run the deterministic two-task demo."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from .demo import build_loop
from .meta import MetaPolicy, MetaSearch


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="g4_"))
    _print_meta(root / "meta")
    loop = build_loop(root)
    result = loop.run_stream(["reconcile", "memo"])
    first, second = result.tasks
    print(f"run directory: {root}")
    print(f"rules unchanged: {result.rules_unchanged}")
    print(f"reconcile adopted: {first.adopted_label}")
    print(f"memo started from base: {second.start_tree == loop.base_tree}")
    print(f"memo adopted: {second.adopted_label}")
    print(f"library entries: {', '.join(loop.library.entries())}")
    kept_skill = "skills/reconcile/SKILL.md" in second.final_tree
    print(f"memo inherited the reconcile skill: {kept_skill}")


def _print_meta(root: Path) -> None:
    search = MetaSearch(root, MetaPolicy(("reopen_check",), 1), heldout_score=0.2, train_score=0.2)
    parent = search.nodes[0]
    broader = MetaPolicy(("reopen_check", "source_reconcile"), 2)
    search.propose(parent, _diff("solution.cpp", "solver"), task_score=0.9)
    search.propose(
        parent,
        _diff("policy.json", "broader"),
        broader,
        screen_score=0.4,
        heldout_score=0.5,
    )
    chosen = search.select_parent(random.Random(0))
    print(f"meta archive: {', '.join(node.gen_id for node in search.nodes)}")
    print(f"task ledger entries: {len(search.task_ledger)}")
    print(f"selected meta parent: {chosen.gen_id} budget {chosen.policy.probe_budget}")
    print(f"meta rules unchanged: {search.rules_unchanged}")


def _diff(path: str, body: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-old\n+{body}\n"
    )


if __name__ == "__main__":
    main()

