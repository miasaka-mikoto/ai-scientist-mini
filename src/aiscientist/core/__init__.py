"""Public core API for AI Scientist Mini.

Everything exported here is offline-safe and UI-independent.  Keeping the
imports central gives CLI, Tkinter and future adapters one stable surface.
"""

from .models import *
from .analysis import *
from .providers import ExperimentProvider, MockLLMExperimentProvider, PythonFunctionProvider, SyntheticProvider
from .store import ResearchStore
from .engine import ApprovalRequired, ExperimentExecution, RuleBasedScientist, ScientificEngine, StopRequested

try:
    from .demo import run_memory_strategy_study
except ImportError:  # demo is added lazily during packaging
    run_memory_strategy_study = None  # type: ignore[assignment]

__all__ = [
    *[name for name in globals() if not name.startswith("_")],
]
