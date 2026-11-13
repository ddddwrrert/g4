"""Fixed stage rules.

The outer loop may update the bandit table and the experience library.
It does not rewrite these strings.
"""

PLANNER_RULES = (
    "Propose one edit for one failure signature. "
    "Keep invariant agent habits separate from domain procedures. "
    "Do not rewrite these instructions."
)

PROBE_RULES = (
    "Run the candidate on the same task as the reference. "
    "Credit only behavior that changed relative to the reference."
)

COMPOSE_RULES = (
    "Keep an edit only when every behavioral check agrees. "
    "Run the merged harness before adopting it. "
    "If checks disagree, adopt an already executed harness."
)
