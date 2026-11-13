"""G4: G3's harness loop plus a HyperAgents-style meta archive.

Task edits and proposal-policy edits are scored on separate ledgers. The
policy archive samples parents with HyperAgents' score_child_prop weights.
Planner, probe, and compose rules stay fixed.
"""

from .loop import ClosedLoop, StreamResult
from .meta import MetaPolicy, MetaSearch

__all__ = ["ClosedLoop", "MetaPolicy", "MetaSearch", "StreamResult"]
