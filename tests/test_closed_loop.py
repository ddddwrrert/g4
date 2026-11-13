"""Acceptance gates for the closed loop. These tests use the scripted executor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from g4.analyze import cluster_failures
from g4.bandit import Bandit
from g4.demo import CatalogEditor, ScriptedExecutor, build_loop
from g4.loop import Direction, Edit, Execution, ProbeRecord
from g4.rules import PLANNER_RULES


class AnalyzerTests(unittest.TestCase):
    def test_duplicate_events_form_one_signature(self) -> None:
        signatures = cluster_failures(["skipped_reopen", "skipped_reopen", "skipped_reconcile"])
        self.assertEqual([item.id for item in signatures], ["output-not-reopened", "sources-not-reconciled"])

    def test_unknown_failure_events_become_signatures(self) -> None:
        signatures = cluster_failures(["skipped_cache", "skipped_reopen"])
        self.assertEqual([item.id for item in signatures], ["output-not-reopened", "skipped-cache"])


class BanditTests(unittest.TestCase):
    def test_two_losses_suppress_a_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bandit = Bandit(path=Path(tmp) / "bandit.json", seed=1)
            self.assertFalse(bandit.suppressed("output-not-reopened", "reopen_check"))
            bandit.update("output-not-reopened", "reopen_check", won=False)
            bandit.update("output-not-reopened", "reopen_check", won=False)
            self.assertTrue(bandit.suppressed("output-not-reopened", "reopen_check"))

    def test_a_win_outranks_an_untried_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bandit = Bandit(path=Path(tmp) / "bandit.json")
            bandit.update("output-not-reopened", "proven", won=True)
            self.assertGreater(
                bandit.rank_key("output-not-reopened", "proven"),
                bandit.rank_key("output-not-reopened", "fresh"),
            )


class StreamTests(unittest.TestCase):
    def test_each_task_starts_from_base_and_reuses_only_a_matching_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rules_before = PLANNER_RULES
            loop = build_loop(Path(tmp))
            result = loop.run_stream(["reconcile", "memo"])
            first, second = result.tasks

            self.assertTrue(result.rules_unchanged)
            self.assertEqual(PLANNER_RULES, rules_before)
            self.assertEqual(first.start_tree, loop.base_tree)
            self.assertEqual(second.start_tree, loop.base_tree)
            self.assertNotEqual(second.start_tree, first.final_tree)
            self.assertEqual(first.adopted_label, "r0-compose")
            self.assertIn("skills/reconcile/SKILL.md", first.final_tree)
            self.assertIn("reopen it and check it", first.final_tree["guardrails.md"])
            self.assertNotIn("skills/reconcile/SKILL.md", second.final_tree)
            self.assertIn("reopen it and check it", second.final_tree["guardrails.md"])
            self.assertTrue(any(item.tree == first.final_tree for item in first.rounds[-1].executions))
            probe_ids = [record.direction.id for record in first.rounds[0].probes]
            self.assertFalse(any("paraphrase" in item for item in probe_ids))
            self.assertGreaterEqual(first.rounds[0].diversity["n_distinct"], 2)

            statuses = {}
            root = loop.library.dir / "harnesses"
            for entry in root.iterdir():
                payload = json.loads((entry / "triple.json").read_text(encoding="utf-8"))
                statuses[payload["signature_id"]] = payload["status"]
            self.assertEqual(statuses["output-not-reopened"], "confirmed")
            self.assertEqual(statuses["sources-not-reconciled"], "pending")
            bandit_text = (loop.library.dir / "bandit.json").read_text(encoding="utf-8")
            self.assertNotIn("Do not rewrite", bandit_text)
            self.assertNotIn(PLANNER_RULES, bandit_text)

    def test_second_round_continues_from_the_executed_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp))
            trace = loop.run_task("reconcile", rounds=2)
            self.assertEqual(trace.rounds[1].start_tree, trace.rounds[0].adopted.tree)
            self.assertNotEqual(trace.rounds[1].start_tree, loop.base_tree)
            self.assertIn("skills/reconcile/SKILL.md", trace.rounds[1].start_tree)

    def test_conflicting_merge_is_executed_and_not_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp))
            trace = loop.run_task("clash")
            labels = [item.label for item in trace.rounds[0].executions]
            self.assertIn("r0-compose", labels)
            self.assertTrue(any(label.startswith("r0-ablate-") for label in labels))
            self.assertNotEqual(trace.adopted_label, "r0-compose")
            restored = {behavior for _, _, behavior in trace.rounds[0].conflicts}
            self.assertEqual(restored, {"reopened_output", "reconciled_sources"})
            stable = [record for record in trace.rounds[0].probes if record.stable]
            self.assertFalse(loop._merge_allowed(stable))
            fresh = ProbeRecord(
                Direction(
                    id="other-reopen",
                    scope="invariant",
                    family="reopen_check",
                    signature_id="output-not-reopened",
                    predicted_behavior="reopened_output",
                    failure_event="skipped_reopen",
                    lesson="A different reopen step.",
                    edit=Edit(appends={"guardrails.md": "\nA different reopen step.\n"}),
                ),
                None,
                {},
                True,
            )
            reconcile = next(record for record in stable if record.direction.family == "source_reconcile")
            self.assertTrue(loop._merge_allowed([fresh, reconcile]))
            self.assertNotIn("reopen it and check it", trace.final_tree["guardrails.md"])
            self.assertTrue(any(item.tree == trace.final_tree for item in trace.rounds[0].executions))

    def test_unstable_leader_is_not_stored(self) -> None:
        class Flaky(ScriptedExecutor):
            def __init__(self) -> None:
                self.seen: dict[str, int] = {}

            def execute(self, task_id: str, tree: dict, label: str) -> Execution:
                key = tree.get("guardrails.md", "")
                self.seen[key] = self.seen.get(key, 0) + 1
                execution = super().execute(task_id, tree, label)
                if "reopen it and check it" in key and self.seen[key] > 1:
                    execution.behaviors = frozenset({"skipped_reopen"})
                    execution.deliverable = "draft"
                return execution

        class ReopenOnly(CatalogEditor):
            def propose(self, task_id, signatures, parent):
                return [
                    item
                    for item in super().propose(task_id, signatures, parent)
                    if item.family == "reopen_check"
                ]

        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp), editor=ReopenOnly(), executor=Flaky())
            trace = loop.run_task("memo")
            self.assertEqual(trace.adopted_label, "r0-reference")
            self.assertEqual(loop.library.entries(), [])

    def test_lower_grade_is_not_a_lesson_and_hindsight_keeps_the_behavior(self) -> None:
        class Cosmetic(CatalogEditor):
            def propose(self, task_id, signatures, parent):
                if not any(item.event == "skipped_reopen" for item in signatures):
                    return []
                return [
                    Direction(
                        id="cosmetic",
                        scope="invariant",
                        family="reopen_check",
                        signature_id="output-not-reopened",
                        predicted_behavior="reopened_output",
                        failure_event="skipped_reopen",
                        lesson="Looking busy is enough.",
                        edit=Edit(
                            appends={
                                "guardrails.md": "\nAfter writing a file, reopen it and check it.\nCOSMETIC\n"
                            }
                        ),
                    )
                ]

        def grade(task_id: str, execution: Execution) -> float:
            del task_id
            if execution.deliverable == "cosmetic":
                return 0.0
            if execution.deliverable == "draft":
                return 0.2
            return 1.0

        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp), editor=Cosmetic(), grader=grade)
            loop.run_task("memo")
            self.assertEqual(loop.library.entries(), [])
            hindsight = (loop.library.dir / "hindsight.jsonl").read_text(encoding="utf-8")
            self.assertIn("reopened_output", hindsight)
            self.assertNotIn("pass_rate", hindsight)

    def test_failed_transfer_is_forgotten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp))
            edit = Edit(appends={"guardrails.md": "\nA note that does not reopen the output.\n"})
            stored = loop.library.add(
                "seed-bad",
                edit.apply(loop.base_tree),
                experience={
                    "cue": "output-not-reopened",
                    "lesson": "A note without a check is enough.",
                    "evidence": "This seed claims a behavior change it does not reproduce.",
                },
                label="seed",
            )
            self.assertIsNotNone(stored)
            assert stored is not None
            (stored / "triple.json").write_text(
                json.dumps(
                    {
                        "signature_id": "output-not-reopened",
                        "family": "reopen_check",
                        "predicted_behavior": "reopened_output",
                        "failure_event": "skipped_reopen",
                        "lesson": "A note without a check is enough.",
                        "status": "pending",
                        **edit.to_json(),
                    }
                ),
                encoding="utf-8",
            )
            trace = loop.run_task("memo")
            self.assertIn("seed-bad", trace.forgotten)
            self.assertNotIn("seed-bad", loop.library.entries())
            self.assertIn("reopen it and check it", trace.final_tree["guardrails.md"])

    def test_library_entry_without_a_triple_is_not_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp))
            fused = dict(loop.base_tree)
            fused["guardrails.md"] += "\nFUSED_MARKER should not be inherited.\n"
            fused["skills/reconcile/SKILL.md"] = "---\nname: reconcile\ndescription: Unused.\n---\n"
            fused["guardrails.md"] += "\nWhen figures must match sources, load the `reconcile` skill.\n"
            loop.library.add(
                "fused-only",
                fused,
                experience={
                    "cue": "output-not-reopened",
                    "lesson": "Keep the whole fused harness.",
                    "evidence": "The fused tree was stored without a signature triple.",
                },
                label="fused",
            )
            trace = loop.run_task("memo")
            self.assertNotIn("FUSED_MARKER", trace.final_tree["guardrails.md"])
            self.assertNotIn("skills/reconcile/SKILL.md", trace.final_tree)
            self.assertIn("fused-only", loop.library.entries())


class QuietEditor:
    def propose(self, task_id, signatures, parent):
        del task_id, signatures, parent
        return []


class StackTests(unittest.TestCase):
    def test_later_retrieval_cannot_erase_an_earlier_behavior(self) -> None:
        class Wipe(ScriptedExecutor):
            def execute(self, task_id: str, tree: dict, label: str) -> Execution:
                execution = super().execute(task_id, tree, label)
                text = tree.get("guardrails.md", "")
                behaviors = set(execution.behaviors)
                if "MARK_A" in text:
                    behaviors.discard("skipped_reopen")
                    behaviors.add("reopened_output")
                if "MARK_B" in text:
                    behaviors.discard("skipped_reconcile")
                    behaviors.add("reconciled_sources")
                if "MARK_A" in text and "MARK_B" in text:
                    behaviors.discard("reopened_output")
                    behaviors.add("skipped_reopen")
                execution.behaviors = frozenset(behaviors)
                tokens = []
                if "reopened_output" in behaviors:
                    tokens.append("rechecked")
                if "reconciled_sources" in behaviors:
                    tokens.append("sources-matched")
                execution.deliverable = " ".join(tokens) if tokens else "draft"
                return execution

        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp), editor=QuietEditor(), executor=Wipe())
            _seed_triple(
                loop,
                "a-keep",
                "output-not-reopened",
                "reopen_check",
                "reopened_output",
                "skipped_reopen",
                Edit(appends={"guardrails.md": "\nMARK_A\n"}),
            )
            _seed_triple(
                loop,
                "b-wipe",
                "sources-not-reconciled",
                "source_reconcile",
                "reconciled_sources",
                "skipped_reconcile",
                Edit(appends={"guardrails.md": "\nMARK_B\n"}),
            )
            trace = loop.run_task("reconcile")
            self.assertIn("MARK_A", trace.final_tree["guardrails.md"])
            self.assertNotIn("MARK_B", trace.final_tree["guardrails.md"])
            self.assertIn("b-wipe", loop.library.entries())
            kept = json.loads((loop.library.entry_dir("a-keep") / "triple.json").read_text(encoding="utf-8"))
            self.assertEqual(kept["status"], "confirmed")

    def test_hindsight_is_proposed_for_the_same_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp), editor=QuietEditor())
            loop.library.dir.mkdir(parents=True, exist_ok=True)
            record = {
                "direction": "missed-reopen",
                "signature_id": "output-not-reopened",
                "family": "reopen_check",
                "predicted_behavior": "reopened_output",
                "failure_event": "skipped_reopen",
                "lesson": "After writing a file, reopen it and check it.",
                "behaviors": ["reopened_output"],
                "errors": ["new_error"],
                "checks": {"ok": False},
                "appends": {"guardrails.md": "\nAfter writing a file, reopen it and check it.\n"},
                "creates": {},
            }
            (loop.library.dir / "hindsight.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            trace = loop.run_task("memo")
            self.assertTrue(any("hindsight-missed-reopen" in record.direction.id for record in trace.rounds[0].probes))
            self.assertIn("reopen it and check it", trace.final_tree["guardrails.md"])

    def test_a_similar_signature_is_tried_without_deleting_the_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp), editor=QuietEditor())
            _seed_triple(
                loop,
                "near-reopen",
                "output-reopened",
                "reopen_check",
                "reopened_output",
                "skipped_reopen",
                Edit(appends={"guardrails.md": "\nAfter writing a file, reopen it and check it.\n"}),
            )
            trace = loop.run_task("memo")
            self.assertNotIn("near-reopen", trace.forgotten)
            self.assertIn("near-reopen", loop.library.entries())
            self.assertIn("reopen it and check it", trace.final_tree["guardrails.md"])

    def test_budget_prefers_the_family_that_has_won(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = build_loop(Path(tmp))
            loop.probe_budget = 1
            loop.bandit.update("output-not-reopened", "proven", won=True)
            chosen = loop._select_directions(
                loop.base_tree,
                [
                    _direction("fresh", "\nFRESH LINE\n"),
                    _direction("proven", "\nPROVEN LINE\n"),
                ],
            )
            self.assertEqual([item.family for item in chosen], ["proven"])


def _direction(family: str, suffix: str) -> Direction:
    return Direction(
        id=family,
        scope="invariant",
        family=family,
        signature_id="output-not-reopened",
        predicted_behavior="reopened_output",
        failure_event="skipped_reopen",
        lesson=family,
        edit=Edit(appends={"guardrails.md": suffix}),
    )


def _seed_triple(loop, entry_id, signature_id, family, predicted, failure, edit: Edit) -> None:
    stored = loop.library.add(
        entry_id,
        edit.apply(loop.base_tree),
        experience={
            "cue": signature_id,
            "lesson": f"Use {entry_id} when {signature_id} appears.",
            "evidence": f"A probe for {entry_id} gained {predicted}.",
        },
        label=entry_id,
    )
    assert stored is not None
    payload = {
        "signature_id": signature_id,
        "family": family,
        "predicted_behavior": predicted,
        "failure_event": failure,
        "lesson": f"Use {entry_id} when {signature_id} appears.",
        "status": "pending",
        **edit.to_json(),
    }
    (stored / "triple.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

