"""Deterministic tasks for the closed loop. No model and no benchmark judge."""

from __future__ import annotations

from pathlib import Path

from .analyze import FailureSignature
from .loop import ClosedLoop, Direction, Edit, Execution

REOPEN = "\nAfter writing a file, reopen it and check it.\n"
TRIGGER = "\nWhen figures must match sources, load the `reconcile` skill.\n"
SKILL = """---
name: reconcile
description: Reconcile figures to source files.
---
Match each figure to its source file.
"""


def base_tree() -> dict[str, str]:
    return {
        "systemprompt.md": "Complete the task with the files that were provided.\n",
        "guardrails.md": "Follow the task. Do not invent source figures.\n",
        "LongTermMEMORY.md": "",
        "tool_descriptions/tool_guidance.md": "Read the files provided with the task.\n",
    }


class CatalogEditor:
    """Offers one invariant edit, one paraphrase of it, and one domain edit."""

    def propose(self, task_id: str, signatures: list[FailureSignature], parent: dict) -> list[Direction]:
        del task_id, parent
        found: list[Direction] = []
        for signature in signatures:
            if signature.event == "skipped_reopen":
                found.append(_reopen(signature, paraphrase=False))
                found.append(_reopen(signature, paraphrase=True))
            elif signature.event == "skipped_reconcile":
                found.append(_reconcile(signature))
        return found


def _reopen(signature: FailureSignature, *, paraphrase: bool) -> Direction:
    suffix = "__paraphrase" if paraphrase else ""
    return Direction(
        id=f"{signature.id}__reopen_check{suffix}",
        scope="invariant",
        family="reopen_check",
        signature_id=signature.id,
        predicted_behavior="reopened_output",
        failure_event="skipped_reopen",
        lesson="After writing a file, reopen it and check it.",
        edit=Edit(appends={"guardrails.md": REOPEN}),
    )


def _reconcile(signature: FailureSignature) -> Direction:
    return Direction(
        id=f"{signature.id}__source_reconcile",
        scope="domain",
        family="source_reconcile",
        signature_id=signature.id,
        predicted_behavior="reconciled_sources",
        failure_event="skipped_reconcile",
        lesson="When figures come from several files, tie each figure back to its source.",
        edit=Edit(
            appends={"guardrails.md": TRIGGER},
            creates={"skills/reconcile/SKILL.md": SKILL},
        ),
    )


class ScriptedExecutor:
    """Behavior is a function of the harness text, so a probe is a real contrast."""

    def execute(self, task_id: str, tree: dict, label: str) -> Execution:
        text = tree.get("guardrails.md", "") + "\n" + tree.get("systemprompt.md", "")
        has_reopen = "reopen it and check it" in text
        has_skill = "skills/reconcile/SKILL.md" in tree and "reconcile" in text
        clash = task_id == "clash" and has_reopen and has_skill
        behaviors: set[str] = set()
        if has_reopen and not clash:
            behaviors.add("reopened_output")
        else:
            behaviors.add("skipped_reopen")
        if task_id != "memo":
            if has_skill and not clash:
                behaviors.add("reconciled_sources")
            else:
                behaviors.add("skipped_reconcile")
        errors: set[str] = set()
        if "INTRODUCE_ERROR" in text:
            errors.add("new_error")
        if "COSMETIC" in text:
            deliverable = "cosmetic"
        else:
            tokens = []
            if "reopened_output" in behaviors:
                tokens.append("rechecked")
            if "reconciled_sources" in behaviors:
                tokens.append("sources-matched")
            deliverable = " ".join(tokens) if tokens else "draft"
        return Execution(
            label=label,
            task_id=task_id,
            tree=dict(tree),
            behaviors=frozenset(behaviors),
            errors=frozenset(errors),
            deliverable=deliverable,
        )


def build_loop(root: Path, **kwargs) -> ClosedLoop:
    root.mkdir(parents=True, exist_ok=True)
    return ClosedLoop(
        editor=kwargs.pop("editor", CatalogEditor()),
        executor=kwargs.pop("executor", ScriptedExecutor()),
        library_root=root / "library",
        base_tree=base_tree(),
        **kwargs,
    )
