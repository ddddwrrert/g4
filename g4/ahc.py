"""AHC measurement against the ORGANON baselines.

Each of the ten problems has its own budget of 100 evaluations. One evaluation
runs seeds 0..49, the same public cases ORGANON's task evaluator uses. The
program sent to the judge starts as that problem's ``solution.cpp``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .demo import base_tree
from .loop import ClosedLoop, Editor, Execution

SessionFactory = Callable[[str, int], Any]

DEFAULT_BUDGET = 100
DEFAULT_SEEDS = 50
DEFAULT_WORKERS = 13


@dataclass(frozen=True)
class AhcProblem:
    id: str
    baseline_path: Path
    statement_path: Path


@dataclass
class AhcScore:
    problem_id: str
    fitness: float
    raw_mean: float
    judge: str
    seed_scores: dict[str, float]
    accepted: bool
    seeds: tuple[int, ...]
    is_baseline: bool
    ran: bool = True
    error: str = ""


class _Budget:
    def __init__(self, max_budget: int) -> None:
        self.max_budget = max_budget
        self.spent = 0

    def charge(self) -> bool:
        if self.spent >= self.max_budget:
            return False
        self.spent += 1
        return True


def default_organon_root() -> Path:
    env = os.environ.get("ORGANON_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "Organon-adversarial"


def load_problems(organon_root: Path) -> tuple[AhcProblem, ...]:
    root = Path(organon_root) / "tasks" / "ale_bench"
    if not root.is_dir():
        raise FileNotFoundError(f"ORGANON ALE-Bench tasks not found: {root}")
    found: list[AhcProblem] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        baseline = path / "solution.cpp"
        evaluator = path / "evaluator" / "evaluator.py"
        if baseline.is_file() and evaluator.is_file():
            found.append(AhcProblem(path.name, baseline, path / "statement.md"))
    if not found:
        raise FileNotFoundError(f"no ALE-Bench baselines under {root}")
    return tuple(found)


def prepare_code(program: str) -> str:
    return program.replace("#EVOLVE_START", "").replace("#EVOLVE_END", "").strip()


def default_session_factory(problem_id: str, num_workers: int) -> Any:
    try:
        from ale_bench_eval.safe_ale_session import start_ale_bench_session
    except ImportError as exc:
        raise RuntimeError(
            "ALE 会话导入失败。安装 ale_bench_eval 的 eval 依赖，或传入 session_factory。"
            f" 原因: {exc}"
        ) from exc
    return start_ale_bench_session(
        problem_id=problem_id,
        lite_version=True,
        num_workers=num_workers,
    )


def _judge_name(result: Any) -> str:
    raw = getattr(result, "overall_judge_result", "")
    return str(getattr(raw, "value", raw) or "")


def _accepted(judge: str) -> bool:
    return judge.split(".")[-1].strip().upper() == "ACCEPTED"


def _optim(session: Any) -> int:
    metadata = getattr(getattr(session, "problem", None), "metadata", None)
    score_type = getattr(metadata, "score_type", "maximize")
    return 1 if score_type == "maximize" else -1


def _score_from_result(
    problem_id: str,
    result: Any,
    seeds: tuple[int, ...],
    optim: int,
    *,
    is_baseline: bool,
) -> AhcScore:
    cases = list(getattr(result, "case_results", []) or [])
    seed_scores = {
        str(seed): float(case.absolute_score)
        for seed, case in zip(seeds, cases)
    }
    raw_sum = float(getattr(result, "overall_absolute_score", 0.0) or 0.0)
    raw_mean = raw_sum / max(1, len(seeds))
    judge = _judge_name(result)
    accepted = _accepted(judge)
    fitness = raw_sum * optim / max(1, len(seeds))
    if not accepted and optim < 0:
        fitness = float(-sys.maxsize - 1)
    return AhcScore(
        problem_id=problem_id,
        fitness=fitness,
        raw_mean=raw_mean,
        judge=judge,
        seed_scores=seed_scores,
        accepted=accepted,
        seeds=seeds,
        is_baseline=is_baseline,
    )


class _Live:
    def __init__(self, session: Any, seeds: tuple[int, ...]) -> None:
        self.session = session
        self.seeds = seeds
        self.cases = list(session.case_gen(list(seeds)))
        self.optim = _optim(session)


class AhcSuite:
    """Ten ORGANON baselines, 100 evaluations each, 50 seeds per evaluation."""

    def __init__(
        self,
        organon_root: Path | None = None,
        *,
        budget: int = DEFAULT_BUDGET,
        n_seeds: int = DEFAULT_SEEDS,
        num_workers: int = DEFAULT_WORKERS,
        session_factory: SessionFactory | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        if n_seeds < 1:
            raise ValueError("n_seeds must be at least 1")
        self.organon_root = Path(organon_root) if organon_root else default_organon_root()
        self.budget = budget
        self.n_seeds = n_seeds
        self.seeds = tuple(range(n_seeds))
        self.num_workers = num_workers
        self.session_factory = session_factory or default_session_factory
        self.problems = load_problems(self.organon_root)
        self._by_id = {problem.id: problem for problem in self.problems}
        self._budgets = {problem.id: _Budget(budget) for problem in self.problems}
        self._sessions: dict[str, _Live] = {}
        self._overrides: dict[str, str] = {}
        self._baseline_text: dict[str, str] = {}
        self._baseline_fitness: dict[str, float] = {}

    @property
    def problem_ids(self) -> list[str]:
        return [problem.id for problem in self.problems]

    def baseline_text(self, problem_id: str) -> str:
        self._require(problem_id)
        if problem_id not in self._baseline_text:
            self._baseline_text[problem_id] = self._by_id[problem_id].baseline_path.read_text(encoding="utf-8")
        return self._baseline_text[problem_id]

    def program(self, problem_id: str) -> str:
        return self._overrides.get(problem_id, self.baseline_text(problem_id))

    def submit(self, problem_id: str, program: str) -> None:
        self._require(problem_id)
        self._overrides[problem_id] = program

    def spent(self, problem_id: str) -> int:
        return self._budgets[self._require(problem_id)].spent

    def remaining(self, problem_id: str) -> int:
        budget = self._budgets[self._require(problem_id)]
        return budget.max_budget - budget.spent

    def baseline_fitness(self, problem_id: str) -> float | None:
        self._require(problem_id)
        return self._baseline_fitness.get(problem_id)

    def evaluate(self, problem_id: str, program: str | None = None) -> AhcScore:
        self._require(problem_id)
        budget = self._budgets[problem_id]
        if not budget.charge():
            return self._not_run(problem_id, f"budget exhausted ({budget.spent}/{budget.max_budget})")
        source = self.program(problem_id) if program is None else program
        prepared = prepare_code(source)
        is_baseline = prepared == prepare_code(self.baseline_text(problem_id))
        try:
            live = self._live(problem_id)
            result = live.session.case_eval(
                live.cases,
                prepared,
                code_language="cpp20",
                skip_local_visualization=True,
                reuse_containers=True,
            )
        except Exception as exc:
            return AhcScore(
                problem_id=problem_id,
                fitness=0.0,
                raw_mean=0.0,
                judge="",
                seed_scores={},
                accepted=False,
                seeds=self.seeds,
                is_baseline=is_baseline,
                error=f"{type(exc).__name__}: {exc}",
            )
        score = _score_from_result(
            problem_id,
            result,
            self.seeds,
            live.optim,
            is_baseline=is_baseline,
        )
        if score.accepted and score.is_baseline:
            self._baseline_fitness.setdefault(problem_id, score.fitness)
        return score

    def grade(self, task_id: str, execution: Execution) -> float | None:
        del task_id
        value = getattr(execution, "fitness", None)
        if value is None:
            return None
        return float(value)

    def close(self) -> None:
        for live in self._sessions.values():
            closer = getattr(live.session, "close", None)
            if closer is not None:
                closer()
        self._sessions.clear()

    def _live(self, problem_id: str) -> _Live:
        live = self._sessions.get(problem_id)
        if live is None:
            session = self.session_factory(problem_id, self.num_workers)
            live = _Live(session, self.seeds)
            self._sessions[problem_id] = live
        return live

    def _require(self, problem_id: str) -> str:
        if problem_id not in self._by_id:
            known = ", ".join(self.problem_ids)
            raise KeyError(f"unknown AHC problem {problem_id!r}; known: {known}")
        return problem_id

    def _not_run(self, problem_id: str, error: str) -> AhcScore:
        return AhcScore(
            problem_id=problem_id,
            fitness=0.0,
            raw_mean=0.0,
            judge="",
            seed_scores={},
            accepted=False,
            seeds=self.seeds,
            is_baseline=False,
            ran=False,
            error=error,
        )


def format_deliverable(score: AhcScore) -> str:
    if not score.ran:
        return "fitness=none"
    return (
        f"problem={score.problem_id}\n"
        f"fitness={score.fitness:.6f}\n"
        f"judge={score.judge}\n"
        f"seeds={','.join(str(seed) for seed in score.seeds)}\n"
    )


def parse_fitness(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("fitness="):
            raw = line.split("=", 1)[1].strip()
            if raw == "none":
                return float("-inf")
            return float(raw)
    return float("-inf")


def fitness_judge(texts: dict[str, str]) -> str:
    """Pick the existing deliverable with the higher AHC fitness."""

    def key(anon: str) -> tuple[float, int, str]:
        return (parse_fitness(texts[anon]), len(texts[anon]), anon)

    return max(texts, key=key)


class AhcExecutor:
    """Score the current C++ program. The harness tree is not the scored artifact."""

    def __init__(self, suite: AhcSuite) -> None:
        self.suite = suite

    def execute(self, task_id: str, tree: Mapping[str, str], label: str) -> Execution:
        if task_id not in self.suite._by_id:
            return Execution(
                label=label,
                task_id=task_id,
                tree=dict(tree),
                behaviors=frozenset({"failed_judge"}),
                errors=frozenset(),
                deliverable="fitness=none",
                ok=False,
                error=f"unknown problem {task_id}",
            )
        score = self.suite.evaluate(task_id)
        if not score.ran:
            return Execution(
                label=label,
                task_id=task_id,
                tree=dict(tree),
                behaviors=frozenset({"failed_budget"}),
                errors=frozenset({"budget"}),
                deliverable=format_deliverable(score),
                ok=False,
                error=score.error,
            )
        behaviors: set[str] = set()
        errors: set[str] = set()
        if score.error:
            behaviors.add("failed_judge")
            errors.add("eval_error")
        elif not score.accepted:
            behaviors.add("failed_judge")
        if len(score.seed_scores) < self.suite.n_seeds:
            behaviors.add("skipped_seed")
        baseline = self.suite.baseline_fitness(task_id)
        if baseline is not None and not score.is_baseline and score.fitness < baseline:
            behaviors.add("failed_regression")
        execution = Execution(
            label=label,
            task_id=task_id,
            tree=dict(tree),
            behaviors=frozenset(behaviors),
            errors=frozenset(errors),
            deliverable=format_deliverable(score),
            ok=True,
            error=score.error,
        )
        execution.fitness = score.fitness  # type: ignore[attr-defined]
        return execution


def build_ahc_loop(
    library_root: Path,
    editor: Editor,
    suite: AhcSuite,
    tree: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> ClosedLoop:
    return ClosedLoop(
        editor=editor,
        executor=AhcExecutor(suite),
        library_root=library_root,
        base_tree=dict(tree) if tree is not None else base_tree(),
        grader=suite.grade,
        judge=fitness_judge,
        **kwargs,
    )


def main() -> None:
    suite = AhcSuite()
    print(f"题目 {len(suite.problems)}")
    print(f"每题预算 {suite.budget}")
    print(f"种子 {suite.seeds[0]}..{suite.seeds[-1]} 共 {suite.n_seeds}")
    for problem in suite.problems:
        print(f"{problem.id}  {problem.baseline_path}")
    try:
        import ale_bench_eval  # noqa: F401
    except Exception as exc:
        print(f"ALE 导入: 不可用 ({type(exc).__name__}: {exc})")
        return
    print("ALE 导入: 可用")


if __name__ == "__main__":
    main()
