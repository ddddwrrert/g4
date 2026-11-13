"""Task loop and cross-task library.

Harness² supplies the editable tree, the evidence-gated library, anonymous
selection ids, and the rollout ledger. This module supplies the gates around
them: cluster failures, probe one edit at a time, rerun leaders, execute a
merge before it can be adopted, and start every new task from the base tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from . import rules
from .analyze import FailureSignature, cluster_failures, signatures_match
from .bandit import Bandit
from .bootstrap import ensure_harness2

ensure_harness2()

from harness2.core.context import assert_tree_has_no_grader_verdicts
from harness2.core.ledger import Ledger
from harness2.core.library import Library, has_experience
from harness2.core.selection import assign_ids
from harness2.core.tree import SurfaceSpec, diff_summary, diversity, validate
from harness2.substrate.opencode import OPENCODE_SURFACE


Tree = dict[str, str]
GradeFn = Callable[[str, "Execution"], float | None]


@dataclass(frozen=True)
class Edit:
    """One serialisable harness change. Retrieval reapplies this, not a fused tree."""

    appends: dict[str, str] = field(default_factory=dict)
    creates: dict[str, str] = field(default_factory=dict)

    def apply(self, tree: Mapping[str, str]) -> Tree:
        out = dict(tree)
        for rel, body in self.creates.items():
            out.setdefault(rel, body)
        for rel, suffix in self.appends.items():
            current = out.get(rel, "")
            if suffix.strip() and suffix.strip() not in current:
                out[rel] = current + suffix
        return out

    def to_json(self) -> dict:
        return {"appends": self.appends, "creates": self.creates}

    @classmethod
    def from_json(cls, payload: Mapping) -> "Edit":
        return cls(
            appends=dict(payload.get("appends") or {}),
            creates=dict(payload.get("creates") or {}),
        )


@dataclass(frozen=True)
class Direction:
    id: str
    scope: str
    family: str
    signature_id: str
    predicted_behavior: str
    failure_event: str
    lesson: str
    edit: Edit


@dataclass
class Execution:
    label: str
    task_id: str
    tree: Tree
    behaviors: frozenset[str]
    errors: frozenset[str]
    deliverable: str
    ok: bool = True
    error: str = ""


class Executor(Protocol):
    def execute(self, task_id: str, tree: Mapping[str, str], label: str) -> Execution: ...


class Editor(Protocol):
    def propose(
        self,
        task_id: str,
        signatures: Sequence[FailureSignature],
        parent: Mapping[str, str],
    ) -> list[Direction]: ...


@dataclass
class ProbeRecord:
    direction: Direction
    execution: Execution | None
    checks: dict[str, bool]
    stable: bool
    invalid: str = ""


@dataclass
class RoundTrace:
    index: int
    start_tree: Tree
    reference: Execution
    probes: list[ProbeRecord]
    executions: list[Execution]
    adopted: Execution
    conflicts: list[tuple[str, str, str]]
    diversity: dict


@dataclass
class TaskTrace:
    task_id: str
    rounds: list[RoundTrace]
    stored: list[str] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)

    @property
    def start_tree(self) -> Tree:
        return self.rounds[0].start_tree

    @property
    def final_tree(self) -> Tree:
        return self.rounds[-1].adopted.tree

    @property
    def adopted_label(self) -> str:
        return self.rounds[-1].adopted.label


@dataclass
class StreamResult:
    tasks: list[TaskTrace]
    rules_unchanged: bool


@dataclass
class _Hit:
    entry_id: str
    signature_id: str
    family: str
    predicted_behavior: str
    failure_event: str
    lesson: str
    edit: Edit
    exact: bool = True


def _copy_tree(tree: Mapping[str, str]) -> Tree:
    return dict(tree)


def _edit_key(edit: Edit) -> str:
    return json.dumps(edit.to_json(), sort_keys=True)


def _contains_edit(edit: Edit, tree: Mapping[str, str]) -> bool:
    for rel, body in edit.creates.items():
        current = tree.get(rel, "")
        if current != body and not current.startswith(body):
            return False
    for rel, suffix in edit.appends.items():
        if suffix.strip() and suffix.strip() not in tree.get(rel, ""):
            return False
    return True


def _fingerprint(surface: SurfaceSpec, parent: Mapping[str, str], direction: Direction) -> tuple:
    child = direction.edit.apply(parent)
    diff = diff_summary(surface, dict(parent), child)
    parts: list[tuple[str, str | None]] = []
    for rel in diff["added"] + diff["modified"]:
        parts.append((rel, child.get(rel)))
    for rel in diff["removed"]:
        parts.append((rel, None))
    return (direction.signature_id, direction.family, tuple(parts))


def _checks(
    task_id: str,
    reference: Execution,
    candidate: Execution,
    direction: Direction,
    grader: GradeFn | None,
) -> dict[str, bool]:
    gained = (
        direction.predicted_behavior in candidate.behaviors
        and direction.predicted_behavior not in reference.behaviors
    )
    failure_gone = direction.failure_event not in candidate.behaviors
    no_new_error = not (set(candidate.errors) - set(reference.errors))
    nonempty = bool(candidate.deliverable.strip()) and candidate.ok
    grade_ok = True
    if grader is not None:
        before = grader(task_id, reference)
        after = grader(task_id, candidate)
        if before is not None and after is not None:
            grade_ok = after >= before
    agreed = gained and failure_gone and no_new_error and nonempty and grade_ok
    return {
        "gained": gained,
        "failure_gone": failure_gone,
        "no_new_error": no_new_error,
        "nonempty": nonempty,
        "grade": grade_ok,
        "ok": agreed,
    }


def blind_select(task_id: str, pool: Sequence[Execution], judge: Callable[[dict[str, str]], str]) -> Execution:
    """Pick an existing deliverable. The judge sees anonymous texts, not harnesses."""
    if not pool:
        raise ValueError("selection pool is empty")
    labels = [item.label for item in pool]
    mapping = assign_ids(labels, task_id=task_id, k=max(1, len(labels)), mode="closed")
    by_label = {item.label: item for item in pool}
    texts = {anon: by_label[label].deliverable for anon, label in mapping.items()}
    chosen = judge(dict(texts))
    if chosen not in mapping:
        chosen = sorted(mapping)[0]
    return by_label[mapping[chosen]]


class ClosedLoop:
    def __init__(
        self,
        *,
        editor: Editor,
        executor: Executor,
        library_root: Path,
        base_tree: Mapping[str, str],
        domain: str = "work",
        surface: SurfaceSpec = OPENCODE_SURFACE,
        seed: int = 0,
        probe_budget: int = 4,
        judge: Callable[[dict[str, str]], str] | None = None,
        grader: GradeFn | None = None,
    ) -> None:
        self.editor = editor
        self.executor = executor
        self.surface = surface
        self.base_tree = _copy_tree(base_tree)
        self.probe_budget = probe_budget
        self.judge = judge or _default_judge
        self.grader = grader
        self.library = Library(root=Path(library_root), domain=domain, surface=surface)
        self.bandit = Bandit(path=self.library.dir / "bandit.json", seed=seed)
        self.bandit.load()
        self._conflicts: set[tuple[str, str]] = set()
        self._last_ablations: list[Execution] = []
        self._load_conflicts()
        self._rules = (rules.PLANNER_RULES, rules.PROBE_RULES, rules.COMPOSE_RULES)
        validate(self.surface, self.base_tree)

    def run_stream(self, task_ids: Sequence[str], *, rounds: int = 1) -> StreamResult:
        traces = [self.run_task(task_id, rounds=rounds) for task_id in task_ids]
        unchanged = (rules.PLANNER_RULES, rules.PROBE_RULES, rules.COMPOSE_RULES) == self._rules
        return StreamResult(tasks=traces, rules_unchanged=unchanged)

    def run_task(self, task_id: str, *, rounds: int = 1, start: Mapping[str, str] | None = None) -> TaskTrace:
        parent = _copy_tree(self.base_tree if start is None else start)
        trace = TaskTrace(task_id=task_id, rounds=[])
        for index in range(rounds):
            parent = self._run_round(trace, parent, index)
        self.bandit.save()
        return trace

    def _run_round(self, trace: TaskTrace, parent: Tree, index: int) -> Tree:
        start = _copy_tree(parent)
        ledger = Ledger(mode="closed", task_id=trace.task_id, k=self.probe_budget)
        origin = self._execute(trace.task_id, start, f"r{index}-reference", ledger)
        reference = origin
        applied: list[_Hit] = []
        if index == 0:
            parent, reference, applied = self._confirm_retrieved(trace, reference, ledger)
        signatures = cluster_failures(reference.behaviors)
        proposals = list(self.editor.propose(trace.task_id, signatures, parent) if signatures else [])
        proposals.extend(self._hindsight_directions(signatures))
        proposals.extend(self._similar_directions(signatures))
        chosen = self._select_directions(parent, proposals)
        probes = [self._probe(trace.task_id, reference, direction, ledger) for direction in chosen]
        stable = [record for record in probes if record.stable]
        executions = [origin]
        if reference is not origin:
            executions.append(reference)
        conflicts: list[tuple[str, str, str]] = []
        merge: Execution | None = None
        if len(stable) >= 2 and self._merge_allowed(stable):
            merged_tree = parent
            for record in stable:
                merged_tree = record.direction.edit.apply(merged_tree)
            merge = self._execute(trace.task_id, merged_tree, f"r{index}-compose", ledger)
            executions.append(merge)
            if not self._merge_keeps(reference, merge, stable):
                conflicts = self._attribute_merge(trace.task_id, parent, merge, stable, ledger, index)
                executions.extend(
                    item
                    for item in self._last_ablations
                    if item not in executions
                )
                merge = None
        pool = [origin]
        if reference is not origin:
            pool.append(reference)
        pool.extend(record.execution for record in stable if record.execution is not None)
        if merge is not None:
            pool.append(merge)
        executions.extend(
            record.execution
            for record in probes
            if record.execution is not None and record.execution not in executions
        )
        adopted = blind_select(trace.task_id, pool, self.judge)
        for hit in applied:
            if _contains_edit(hit.edit, adopted.tree):
                self._mark(hit.entry_id, "confirmed")
        for record in stable:
            if record.execution is None or not _contains_edit(record.direction.edit, adopted.tree):
                continue
            entry = self._retain(trace.task_id, record)
            if entry:
                trace.stored.append(entry)
        self._hindsight(reference, probes)
        trace.rounds.append(
            RoundTrace(
                index=index,
                start_tree=start,
                reference=reference,
                probes=probes,
                executions=executions,
                adopted=adopted,
                conflicts=conflicts,
                diversity=diversity(
                    {
                        record.direction.id: record.execution.tree
                        for record in probes
                        if record.execution is not None
                    }
                ),
            )
        )
        return _copy_tree(adopted.tree)

    def _confirm_retrieved(
        self, trace: TaskTrace, reference: Execution, ledger: Ledger
    ) -> tuple[Tree, Execution, list[_Hit]]:
        current_tree = reference.tree
        current = reference
        protected: set[str] = set()
        applied: list[_Hit] = []
        for hit in self._hits_for(cluster_failures(reference.behaviors), exact=True):
            trial = hit.edit.apply(current_tree)
            execution = self._execute(trace.task_id, trial, f"transfer-{hit.entry_id}", ledger)
            direction = self._direction_from_hit(hit)
            agreed = _checks(trace.task_id, current, execution, direction, self.grader)["ok"]
            kept_previous = protected <= set(execution.behaviors)
            if agreed and kept_previous:
                self.bandit.update(hit.signature_id, hit.family, won=True)
                protected.add(hit.predicted_behavior)
                applied.append(hit)
                current_tree = trial
                current = execution
            elif not agreed:
                self.library.forget([hit.entry_id])
                self._mark_forgotten(trace, hit.entry_id)
                self.bandit.update(hit.signature_id, hit.family, won=False)
            else:
                self._remember_edit_conflict(direction, applied, execution)
        return current_tree, current, applied

    def _direction_from_hit(self, hit: _Hit, signature_id: str | None = None) -> Direction:
        return Direction(
            id=hit.entry_id,
            scope="retrieved",
            family=hit.family,
            signature_id=signature_id or hit.signature_id,
            predicted_behavior=hit.predicted_behavior,
            failure_event=hit.failure_event,
            lesson=hit.lesson,
            edit=hit.edit,
        )

    def _select_directions(self, parent: Mapping[str, str], proposals: Sequence[Direction]) -> list[Direction]:
        seen: set[tuple] = set()
        ranked: list[tuple[float, Direction]] = []
        for direction in proposals:
            if self.bandit.suppressed(direction.signature_id, direction.family):
                continue
            fingerprint = _fingerprint(self.surface, parent, direction)
            if fingerprint in seen or fingerprint[2] == ():
                continue
            seen.add(fingerprint)
            ranked.append(
                (self.bandit.rank_key(direction.signature_id, direction.family), direction)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [direction for _, direction in ranked[: self.probe_budget]]

    def _probe(
        self, task_id: str, reference: Execution, direction: Direction, ledger: Ledger
    ) -> ProbeRecord:
        try:
            child = direction.edit.apply(reference.tree)
            validate(self.surface, child)
            assert_tree_has_no_grader_verdicts(child)
        except ValueError as exc:
            self.bandit.update(direction.signature_id, direction.family, won=False)
            return ProbeRecord(direction, None, {"ok": False}, False, invalid=str(exc))
        first = self._execute(task_id, child, f"probe-{direction.id}", ledger)
        checks = _checks(task_id, reference, first, direction, self.grader)
        stable = bool(checks["ok"])
        execution = first
        if stable:
            second = self._execute(task_id, child, f"confirm-{direction.id}", ledger)
            again = _checks(task_id, reference, second, direction, self.grader)
            stable = bool(again["ok"])
            checks = again
            execution = second if stable else first
        self.bandit.update(direction.signature_id, direction.family, won=stable)
        return ProbeRecord(direction, execution, checks, stable)

    def _merge_allowed(self, stable: Sequence[ProbeRecord]) -> bool:
        keys = [_edit_key(record.direction.edit) for record in stable]
        for left, right in self._conflicts:
            if left in keys and right in keys:
                return False
        return True

    def _merge_keeps(
        self, reference: Execution, merge: Execution, stable: Sequence[ProbeRecord]
    ) -> bool:
        if not merge.ok or (set(merge.errors) - set(reference.errors)):
            return False
        return all(record.direction.predicted_behavior in merge.behaviors for record in stable)

    def _attribute_merge(
        self,
        task_id: str,
        parent: Tree,
        merge: Execution,
        stable: Sequence[ProbeRecord],
        ledger: Ledger,
        index: int,
    ) -> list[tuple[str, str, str]]:
        """Drop one edit at a time. A behavior that returns was suppressed by the edit that was dropped."""
        lost = [record for record in stable if record.direction.predicted_behavior not in merge.behaviors]
        self._last_ablations = []
        found: list[tuple[str, str, str]] = []
        for removed in stable:
            trial = parent
            for other in stable:
                if other.direction.id != removed.direction.id:
                    trial = other.direction.edit.apply(trial)
            execution = self._execute(
                task_id, trial, f"r{index}-ablate-{removed.direction.id}", ledger
            )
            self._last_ablations.append(execution)
            for victim in lost:
                behavior = victim.direction.predicted_behavior
                if behavior in execution.behaviors and behavior not in merge.behaviors:
                    found.append((removed.direction.id, victim.direction.id, behavior))
                    self._remember_pair(removed.direction.edit, victim.direction.edit, found[-1])
        return found

    def _remember_pair(self, suppressor: Edit, victim: Edit, detail: tuple[str, str, str]) -> None:
        pair = (_edit_key(suppressor), _edit_key(victim))
        if pair in self._conflicts:
            return
        self._conflicts.add(pair)
        path = self.library.dir / "conflicts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "suppressor": detail[0],
                        "victim": detail[1],
                        "behavior": detail[2],
                        "suppressor_edit": suppressor.to_json(),
                        "victim_edit": victim.to_json(),
                    }
                )
                + "\n"
            )

    def _remember_edit_conflict(self, direction: Direction, applied: Sequence[_Hit], execution: Execution) -> None:
        for hit in applied:
            if hit.predicted_behavior in execution.behaviors:
                continue
            self._remember_pair(
                direction.edit,
                hit.edit,
                (direction.id, hit.entry_id, hit.predicted_behavior),
            )

    def _retain(self, task_id: str, record: ProbeRecord) -> str:
        direction = record.direction
        if record.execution is None:
            return ""
        entry_id = f"{task_id}__{direction.id}"
        if (self.library.entry_dir(entry_id) / "triple.json").is_file():
            return ""
        single = direction.edit.apply(self.base_tree)
        try:
            validate(self.surface, single)
        except ValueError:
            return ""
        diff = diff_summary(self.surface, self.base_tree, single)
        experience = {
            "cue": direction.signature_id,
            "lesson": direction.lesson,
            "evidence": (
                f"Probe {record.execution.label} gained {direction.predicted_behavior} "
                f"and cleared {direction.failure_event}. "
                f"Changed files: {', '.join(diff['added'] + diff['modified']) or 'none'}."
            ),
            "confidence": "pending until another task shows the same behavior",
        }
        if not has_experience(experience):
            return ""
        stored = self.library.add(entry_id, single, experience=experience, label=record.execution.label)
        if stored is None:
            return ""
        payload = {
            "signature_id": direction.signature_id,
            "family": direction.family,
            "scope": direction.scope,
            "predicted_behavior": direction.predicted_behavior,
            "failure_event": direction.failure_event,
            "lesson": direction.lesson,
            "status": "pending",
            **direction.edit.to_json(),
        }
        (stored / "triple.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return entry_id

    def _hindsight(self, reference: Execution, probes: Sequence[ProbeRecord]) -> None:
        path = self.library.dir / "hindsight.jsonl"
        lines = []
        for record in probes:
            execution = record.execution
            if execution is None or record.stable:
                continue
            if execution.behaviors == reference.behaviors and execution.errors == reference.errors:
                continue
            lines.append(
                json.dumps(
                    {
                        "direction": record.direction.id,
                        "signature_id": record.direction.signature_id,
                        "family": record.direction.family,
                        "predicted_behavior": record.direction.predicted_behavior,
                        "failure_event": record.direction.failure_event,
                        "lesson": record.direction.lesson,
                        "behaviors": sorted(execution.behaviors),
                        "errors": sorted(execution.errors),
                        "checks": record.checks,
                        **record.direction.edit.to_json(),
                    }
                )
            )
        if not lines:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _hits_for(self, signatures: Sequence[FailureSignature], *, exact: bool) -> list[_Hit]:
        hits: list[_Hit] = []
        root = self.library.dir / "harnesses"
        if not root.is_dir():
            return []
        for entry in sorted(path for path in root.iterdir() if path.is_dir()):
            triple_path = entry / "triple.json"
            if not triple_path.is_file():
                continue
            payload = json.loads(triple_path.read_text(encoding="utf-8"))
            if payload.get("status") == "revoked":
                continue
            stored = str(payload.get("signature_id") or "")
            matched = self._matching_signature(signatures, stored, exact=exact)
            if matched is None:
                continue
            hits.append(
                _Hit(
                    entry_id=entry.name,
                    signature_id=stored,
                    family=str(payload.get("family") or ""),
                    predicted_behavior=str(payload["predicted_behavior"]),
                    failure_event=str(payload["failure_event"]),
                    lesson=str(payload.get("lesson") or ""),
                    edit=Edit.from_json(payload),
                    exact=stored == matched.id,
                )
            )
        return hits

    def _matching_signature(
        self, signatures: Sequence[FailureSignature], stored: str, *, exact: bool
    ) -> FailureSignature | None:
        for signature in signatures:
            same = stored == signature.id or stored == signature.event
            close = signatures_match(stored, signature.id) or signatures_match(stored, signature.event)
            if same or (not exact and close):
                return signature
        return None

    def _similar_directions(self, signatures: Sequence[FailureSignature]) -> list[Direction]:
        found: list[Direction] = []
        for hit in self._hits_for(signatures, exact=False):
            if hit.exact:
                continue
            matched = self._matching_signature(signatures, hit.signature_id, exact=False)
            if matched is None:
                continue
            found.append(self._direction_from_hit(hit, matched.id))
        return found

    def _hindsight_directions(self, signatures: Sequence[FailureSignature]) -> list[Direction]:
        path = self.library.dir / "hindsight.jsonl"
        if not path.is_file() or not signatures:
            return []
        found: list[Direction] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            stored = str(payload.get("signature_id") or "")
            failure_event = str(payload.get("failure_event") or "")
            matched = self._matching_signature(signatures, stored, exact=False)
            if matched is None and not any(signature.event == failure_event for signature in signatures):
                continue
            if matched is None:
                matched = next(signature for signature in signatures if signature.event == failure_event)
            if "appends" not in payload and "creates" not in payload:
                continue
            found.append(
                Direction(
                    id=f"hindsight-{payload.get('direction', stored)}",
                    scope="hindsight",
                    family=str(payload.get("family") or "hindsight"),
                    signature_id=matched.id,
                    predicted_behavior=str(payload.get("predicted_behavior") or ""),
                    failure_event=failure_event,
                    lesson=str(payload.get("lesson") or ""),
                    edit=Edit.from_json(payload),
                )
            )
        return found

    def _mark(self, entry_id: str, status: str) -> None:
        path = self.library.entry_dir(entry_id) / "triple.json"
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _mark_forgotten(self, trace: TaskTrace, entry_id: str) -> None:
        if entry_id not in trace.forgotten:
            trace.forgotten.append(entry_id)

    def _load_conflicts(self) -> None:
        path = self.library.dir / "conflicts.jsonl"
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            suppressor = payload.get("suppressor_edit")
            victim = payload.get("victim_edit")
            if not suppressor or not victim:
                continue
            self._conflicts.add((_edit_key(Edit.from_json(suppressor)), _edit_key(Edit.from_json(victim))))

    def _execute(self, task_id: str, tree: Mapping[str, str], label: str, ledger: Ledger) -> Execution:
        try:
            execution = self.executor.execute(task_id, _copy_tree(tree), label)
        except Exception as exc:
            execution = Execution(
                label=label,
                task_id=task_id,
                tree=_copy_tree(tree),
                behaviors=frozenset(),
                errors=frozenset({"executor_error"}),
                deliverable="",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        execution.tree = _copy_tree(execution.tree or tree)
        execution.label = label
        ledger.rollout(label)
        return execution


def _default_judge(texts: dict[str, str]) -> str:
    def score(body: str) -> tuple[int, int]:
        return (body.count("rechecked") + body.count("sources-matched"), len(body))

    return max(sorted(texts), key=lambda anon: score(texts[anon]))
