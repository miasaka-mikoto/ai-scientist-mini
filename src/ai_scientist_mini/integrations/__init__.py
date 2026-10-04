"""Stable integration boundary for AI Scientist Mini.

The integration package deliberately speaks in JSON-compatible records and
small protocols.  It does not import any of the future projects (Paper2Lab,
LLM Lab, Agent Arena, or Synthetic Benchmark Factory).  Those projects can
connect through one of the adapters in :mod:`ai_scientist_mini.integrations.adapters`
or by implementing :class:`ExperimentProvider` themselves.

The rest of the application can therefore evolve without coupling its domain
objects to an external project's Python package.
"""

from .contracts import (
    API_VERSION,
    AdapterDescriptor,
    ExperimentProvider,
    ExperimentRequest,
    ExperimentResult,
    ProviderCapabilities,
    RunResult,
)
from .adapters import (
    AdapterRegistry,
    HttpJsonAdapter,
    JsonLineAdapter,
    LegacyProviderAdapter,
    LocalProviderAdapter,
    load_adapter_manifest,
)
from .exchange import (
    canonical_json,
    export_json,
    export_experiment_request,
    export_experiment_result,
    export_study_state,
    import_envelope,
    import_json,
    import_study_state,
    make_envelope,
    parse_envelope,
    stable_hash,
)
from .api import IntegrationService, create_api_server, serve_api

__all__ = [
    "API_VERSION",
    "AdapterDescriptor",
    "AdapterRegistry",
    "ExperimentProvider",
    "ExperimentRequest",
    "ExperimentResult",
    "HttpJsonAdapter",
    "IntegrationService",
    "JsonLineAdapter",
    "LegacyProviderAdapter",
    "LocalProviderAdapter",
    "load_adapter_manifest",
    "ProviderCapabilities",
    "RunResult",
    "canonical_json",
    "create_api_server",
    "export_experiment_request",
    "export_experiment_result",
    "export_json",
    "export_study_state",
    "import_envelope",
    "import_json",
    "import_study_state",
    "make_envelope",
    "parse_envelope",
    "serve_api",
    "stable_hash",
]
