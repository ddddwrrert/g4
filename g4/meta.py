"""Meta-strategy archive for AHC.

The mutable object is a proposal policy: which families to try, and how many
probes to spend. Planner, probe, and compose rules stay the constants in
rules.py. First-order edits change task code and are scored on their own
ledger. Second-order edits change the policy. A second-order child becomes a
parent only when its held-out score is higher than its parent's. Parent
sampling uses HyperAgents' score_child_prop weights on those held-out scores.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from . import rules

_HYPER = Path(__file__).resolve().parents[1] / "third_party" / "hyperagents"
if str(_HYPER) not in sys.path:
    sys.path.insert(0, str(_HYPER))

from selection import choose_parent, filter_patch_by_files  # noqa: E402

TASK_FILE = "solution.cpp"
META_FILE = "policy.json"


@dataclass(frozen=True)
class MetaPolicy:
    """Proposal policy. This is the only object the outer population edits."""

    families: tuple[str, ...]
    probe_budget: int
    rank_scale: float = 1.0

    def fingerprint(self) -> str:
        payload = {
            "families": list(self.families),
            "probe_budget": self.probe_budget,
            "rank_scale": self.rank_scale,
        }
        return json.dumps(payload, sort_keys=True)

    def to_json(self) -> dict:
        return json.loads(self.fingerprint())

    @classmethod
    def from_json(cls, payload: dict) -> "MetaPolicy":
        return cls(
            families=tuple(payload["families"]),
            probe_budget=int(payload["probe_budget"]),
            rank_scale=float(payload.get("rank_scale", 1.0)),
        )


@dataclass
class MetaNode:
    gen_id: str
    parent_id: str | None
    policy: MetaPolicy
    heldout_score: float | None
    train_score: float | None
    valid_parent: bool


@dataclass(frozen=True)
class Decision:
    order: str
    admitted: bool
    reason: str
    gen_id: str = ""


class PolicyEditor:
    """Keep only directions whose family is in the selected policy."""

    def __init__(self, inner, policy: MetaPolicy) -> None:
        self.inner = inner
        self.policy = policy

    def propose(self, task_id: str, signatures: Sequence, parent: dict) -> list:
        allowed = set(self.policy.families)
        return [item for item in self.inner.propose(task_id, signatures, parent) if item.family in allowed]


def apply_policy(loop, policy: MetaPolicy) -> None:
    """Point a task loop at this policy's probe budget. Rules are not written."""

    loop.probe_budget = policy.probe_budget


def classify_patch(patch: str) -> str:
    """Separate task-code diffs from policy diffs. A diff that touches both is mixed."""

    task = _touches(patch, TASK_FILE)
    meta = _touches(patch, META_FILE)
    if task and meta:
        return "mixed"
    if task:
        return "first"
    if meta:
        return "second"
    return "empty"


def _touches(patch: str, filename: str) -> bool:
    header = f"a/{filename}"
    if header not in patch or f"b/{filename}" not in patch:
        return False
    filtered = filter_patch_by_files(patch, [filename])
    return filtered.strip() != patch.strip()


class MetaSearch:
    def __init__(
        self,
        root: Path,
        base: MetaPolicy,
        heldout_score: float,
        train_score: float,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes: list[MetaNode] = []
        self.task_ledger: list[dict] = []
        self._rules = (rules.PLANNER_RULES, rules.PROBE_RULES, rules.COMPOSE_RULES)
        self._next = 0
        self._add(
            MetaNode(
                gen_id=self._gen_id(),
                parent_id=None,
                policy=base,
                heldout_score=heldout_score,
                train_score=train_score,
                valid_parent=True,
            )
        )

    @property
    def rules_unchanged(self) -> bool:
        return self._rules == (rules.PLANNER_RULES, rules.PROBE_RULES, rules.COMPOSE_RULES)

    def select_parent(self, rng: random.Random) -> MetaNode:
        eligible = [node for node in self.nodes if node.valid_parent and node.heldout_score is not None]
        if not eligible:
            raise RuntimeError("archive has no selectable meta policy")
        counts = [self._child_count(node.gen_id) for node in eligible]
        chosen = choose_parent(
            [node.gen_id for node in eligible],
            [float(node.heldout_score) for node in eligible],
            counts,
            rng,
        )
        return self._by_id(chosen)

    def propose(
        self,
        parent: MetaNode,
        patch: str,
        policy: MetaPolicy | None = None,
        *,
        screen_score: float | None = None,
        heldout_score: float | None = None,
        task_score: float | None = None,
        heldout_fn: Callable[[], float] | None = None,
    ) -> Decision:
        order = classify_patch(patch)
        if order in {"mixed", "empty"}:
            return Decision(order, False, "split")
        if order == "first":
            self.task_ledger.append(
                {
                    "parent_id": parent.gen_id,
                    "task_score": task_score,
                    "order": "first",
                }
            )
            self._flush_task_ledger()
            return Decision("first", False, "task-ledger")
        if policy is None:
            return Decision("second", False, "missing-policy")
        if any(node.policy.fingerprint() == policy.fingerprint() for node in self.nodes):
            return Decision("second", False, "duplicate")
        screen_ok = screen_score is not None and (
            parent.train_score is None or screen_score >= parent.train_score
        )
        if not screen_ok:
            node = self._child(parent, policy, heldout_score=None, train_score=screen_score, valid=False)
            return Decision("second", False, "screen", node.gen_id)
        measured = heldout_fn() if heldout_fn is not None else heldout_score
        improved = (
            measured is not None
            and parent.heldout_score is not None
            and measured > parent.heldout_score
        )
        node = self._child(
            parent,
            policy,
            heldout_score=measured,
            train_score=screen_score,
            valid=improved,
        )
        return Decision("second", improved, "heldout" if improved else "heldout-not-better", node.gen_id)

    def _child(
        self,
        parent: MetaNode,
        policy: MetaPolicy,
        *,
        heldout_score: float | None,
        train_score: float | None,
        valid: bool,
    ) -> MetaNode:
        node = MetaNode(
            gen_id=self._gen_id(),
            parent_id=parent.gen_id,
            policy=policy,
            heldout_score=heldout_score,
            train_score=train_score,
            valid_parent=valid,
        )
        self._add(node)
        return node

    def _add(self, node: MetaNode) -> None:
        self.nodes.append(node)
        archive_ids = [item.gen_id for item in self.nodes]
        line = {"current_genid": node.gen_id, "archive": archive_ids}
        with (self.root / "archive.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        record = {
            "gen_id": node.gen_id,
            "parent_id": node.parent_id,
            "policy": node.policy.to_json(),
            "heldout_score": node.heldout_score,
            "train_score": node.train_score,
            "valid_parent": node.valid_parent,
        }
        with (self.root / "nodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _flush_task_ledger(self) -> None:
        path = self.root / "task_ledger.jsonl"
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in self.task_ledger),
            encoding="utf-8",
        )

    def _child_count(self, gen_id: str) -> int:
        return sum(1 for node in self.nodes if node.parent_id == gen_id)

    def _by_id(self, gen_id: str) -> MetaNode:
        for node in self.nodes:
            if node.gen_id == gen_id:
                return node
        raise KeyError(gen_id)

    def _gen_id(self) -> str:
        gen_id = f"g{self._next}"
        self._next += 1
        return gen_id
