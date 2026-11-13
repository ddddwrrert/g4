"""Contextual bandit over direction families.

The context is a failure signature. The arm is a direction family.
Wins and losses come from probes, not from a rewritten planner prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _variance(wins: int, losses: int) -> float:
    alpha = wins + 1
    beta = losses + 1
    total = alpha + beta
    return (alpha * beta) / (total * total * (total + 1))


@dataclass
class Bandit:
    path: Path
    seed: int = 0
    counts: dict[str, list[int]] = field(default_factory=dict)

    def load(self) -> None:
        if self.path.is_file():
            self.counts = {
                key: [int(pair[0]), int(pair[1])]
                for key, pair in json.loads(self.path.read_text(encoding="utf-8")).items()
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: [wins, losses] for key, (wins, losses) in self.counts.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _peek(self, signature_id: str, family: str) -> tuple[int, int]:
        wins, losses = self.counts.get(f"{signature_id}|{family}", [0, 0])
        return int(wins), int(losses)

    def update(self, signature_id: str, family: str, *, won: bool) -> None:
        key = f"{signature_id}|{family}"
        wins, losses = self._peek(signature_id, family)
        if won:
            wins += 1
        else:
            losses += 1
        self.counts[key] = [wins, losses]

    def suppressed(self, signature_id: str, family: str) -> bool:
        wins, losses = self._peek(signature_id, family)
        pulls = wins + losses
        if pulls < 2:
            return False
        return wins / pulls < 0.35

    def uncertainty(self, signature_id: str, family: str) -> float:
        wins, losses = self._peek(signature_id, family)
        return _variance(wins, losses)

    def rank_key(self, signature_id: str, family: str) -> float:
        """Posterior mean, plus a small bonus for arms that are still uncertain.

        A win moves the arm ahead of an untried one. A loss moves it behind.
        """
        wins, losses = self._peek(signature_id, family)
        mean = (wins + 1) / (wins + losses + 2)
        return mean + 0.1 * self.uncertainty(signature_id, family)
