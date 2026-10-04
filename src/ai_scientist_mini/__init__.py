"""AI Scientist Mini — a local, auditable scientific-loop sandbox."""

# Keep the runtime version in one small, dependency-free place so the CLI,
# packaged executable, and integration envelopes can report the same value.
__version__ = "0.1.0"

from .analysis import ResultAnalyzer
from .engine import (
    ApprovalRequired,
    DuplicateExperiment,
    ScientistEngine,
    StopRequested,
    design_fingerprint,
)
from .models import *  # noqa: F401,F403 - public domain API
from .persistence import (
    JsonProjectStore,
    ProjectRepository,
    SQLiteProjectStore,
    load_project,
    load_sqlite,
    save_project,
    save_sqlite,
)
from .scientist import RuleBasedScientist
from .providers import (
    ExperimentProvider,
    MockLLMExperimentProvider,
    ProviderError,
    ProviderRegistry,
    ProviderResult,
    PythonFunctionProvider,
    SyntheticProvider,
)
# Versioned JSON/CLI/API integration boundary.  These imports are lightweight
# and do not load any future project or network client.
from .integrations import (
    AdapterRegistry,
    ExperimentRequest,
    ExperimentResult,
    HttpJsonAdapter,
    JsonLineAdapter,
    LegacyProviderAdapter,
    LocalProviderAdapter,
    ProviderCapabilities,
    RunResult,
    export_json,
    import_json,
    stable_hash,
)

__all__ = [
    "__version__",
    "ScientistEngine",
    "ResultAnalyzer",
    "SyntheticProvider",
    "PythonFunctionProvider",
    "MockLLMExperimentProvider",
    "ExperimentProvider",
    "ProviderRegistry",
    "ProviderResult",
    "ProviderError",
    "JsonProjectStore",
    "ProjectRepository",
    "SQLiteProjectStore",
    "save_project",
    "load_project",
    "save_sqlite",
    "load_sqlite",
    "DuplicateExperiment",
    "ApprovalRequired",
    "StopRequested",
    "design_fingerprint",
    "RuleBasedScientist",
    "AdapterRegistry",
    "ExperimentRequest",
    "ExperimentResult",
    "HttpJsonAdapter",
    "JsonLineAdapter",
    "LegacyProviderAdapter",
    "LocalProviderAdapter",
    "ProviderCapabilities",
    "RunResult",
    "export_json",
    "import_json",
    "stable_hash",
]
