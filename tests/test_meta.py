"""G4 meta archive: HyperAgents parent weights, split ledgers, held-out admission."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from g4.demo import CatalogEditor
from g4.meta import MetaPolicy, MetaSearch, PolicyEditor, apply_policy, classify_patch
from g4.rules import COMPOSE_RULES, PLANNER_RULES, PROBE_RULES
from selection import score_child_prop_weights


def _patch(path: str, body: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-old\n+{body}\n"
    )


class WeightTests(unittest.TestCase):
    def test_many_children_lose_to_an_equal_score(self) -> None:
        fresh, crowded = score_child_prop_weights([0.5, 0.5], [0, 40])
        self.assertGreater(fresh, 0.99)
        self.assertLess(crowded, 0.01)

    def test_higher_heldout_score_outweighs_an_equal_child_count(self) -> None:
        strong, weak = score_child_prop_weights([0.9, 0.1], [0, 0])
        self.assertGreater(strong, weak)


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="g4_meta_"))
        self.search = MetaSearch(
            self.root,
            MetaPolicy(("reopen_check",), probe_budget=1),
            heldout_score=0.2,
            train_score=0.2,
        )
        self.parent = self.search.nodes[0]

    def test_mixed_patch_is_not_scored(self) -> None:
        patch = _patch("solution.cpp", "task") + _patch("policy.json", "meta")
        decision = self.search.propose(
            self.parent,
            patch,
            MetaPolicy(("reopen_check",), 4),
            screen_score=1.0,
            heldout_score=1.0,
            task_score=1.0,
        )
        self.assertEqual(decision.order, "mixed")
        self.assertEqual(len(self.search.nodes), 1)
        self.assertEqual(self.search.task_ledger, [])

    def test_first_order_stays_off_the_meta_archive(self) -> None:
        decision = self.search.propose(
            self.parent,
            _patch("solution.cpp", "solver"),
            task_score=0.8,
        )
        self.assertEqual(decision.reason, "task-ledger")
        self.assertEqual([node.gen_id for node in self.search.nodes], ["g0"])
        self.assertEqual(self.search.task_ledger[0]["task_score"], 0.8)
        chosen = self.search.select_parent(random.Random(0))
        self.assertEqual(chosen.gen_id, "g0")

    def test_train_only_gain_is_not_a_parent(self) -> None:
        policy = MetaPolicy(("source_reconcile",), probe_budget=2)
        decision = self.search.propose(
            self.parent,
            _patch("policy.json", policy.fingerprint()),
            policy,
            screen_score=0.9,
            heldout_score=0.2,
        )
        self.assertFalse(decision.admitted)
        self.assertFalse(self.search.nodes[-1].valid_parent)
        self.assertEqual(self.search.select_parent(random.Random(1)).gen_id, "g0")

    def test_heldout_gain_joins_the_archive(self) -> None:
        policy = MetaPolicy(("reopen_check", "source_reconcile"), probe_budget=2)
        decision = self.search.propose(
            self.parent,
            _patch("policy.json", "broader"),
            policy,
            screen_score=0.4,
            heldout_score=0.5,
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(self.search.nodes[-1].heldout_score, 0.5)
        weights_parent = score_child_prop_weights([0.2, 0.5], [1, 0])
        self.assertGreater(weights_parent[1], weights_parent[0])

    def test_failed_screen_does_not_read_heldout(self) -> None:
        calls: list[str] = []

        def heldout() -> float:
            calls.append("heldout")
            return 1.0

        decision = self.search.propose(
            self.parent,
            _patch("policy.json", "early"),
            MetaPolicy(("source_reconcile",), 3),
            screen_score=0.0,
            heldout_fn=heldout,
        )
        self.assertEqual(decision.reason, "screen")
        self.assertEqual(calls, [])
        self.assertIsNone(self.search.nodes[-1].heldout_score)

    def test_duplicate_policy_is_not_stored_again(self) -> None:
        policy = MetaPolicy(("source_reconcile",), 2)
        patch = _patch("policy.json", "once")
        self.search.propose(self.parent, patch, policy, screen_score=0.4, heldout_score=0.6)
        again = self.search.propose(self.parent, patch, policy, screen_score=0.4, heldout_score=0.9)
        self.assertEqual(again.reason, "duplicate")
        self.assertEqual(len(self.search.nodes), 2)

    def test_rules_stay_fixed_and_budget_follows_the_policy(self) -> None:
        policy = MetaPolicy(("reopen_check",), probe_budget=2)

        class Loop:
            probe_budget = 4

        loop = Loop()
        apply_policy(loop, policy)
        editor = PolicyEditor(CatalogEditor(), policy)
        proposed = editor.propose("reconcile", _signatures(), {})
        self.assertEqual(loop.probe_budget, 2)
        self.assertTrue(all(item.family == "reopen_check" for item in proposed))
        self.assertEqual(
            (PLANNER_RULES, PROBE_RULES, COMPOSE_RULES),
            self.search._rules,
        )
        self.assertTrue(self.search.rules_unchanged)

    def test_archive_jsonl_records_every_generation(self) -> None:
        self.search.propose(
            self.parent,
            _patch("policy.json", "kept"),
            MetaPolicy(("source_reconcile",), 2),
            screen_score=0.4,
            heldout_score=0.7,
        )
        lines = (self.root / "archive.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('"archive": ["g0", "g1"]', lines[-1])


def _signatures():
    from g4.analyze import cluster_failures

    return cluster_failures(["skipped_reopen", "skipped_reconcile"])


class ClassifyTests(unittest.TestCase):
    def test_orders(self) -> None:
        self.assertEqual(classify_patch(_patch("solution.cpp", "a")), "first")
        self.assertEqual(classify_patch(_patch("policy.json", "b")), "second")
        self.assertEqual(classify_patch("not a diff\n"), "empty")


if __name__ == "__main__":
    unittest.main()
