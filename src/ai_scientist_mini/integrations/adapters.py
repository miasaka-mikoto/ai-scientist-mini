"""Adapters and registry for local, JSON-line, and HTTP providers.

The adapters form a narrow process boundary.  A future project only needs to
implement the JSON contract; it does not need to be installed in the AI
Scientist Mini process.  External adapters are opt-in and never bypass the
host application's budget or approval checks.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import (
    API_VERSION,
    AdapterDescriptor,
    ExperimentProvider,
    ExperimentRequest,
    ExperimentResult,
    ProviderCapabilities,
    RunResult,
    provider_to_descriptor,
)


class AdapterError(RuntimeError):
    """A transport or contract error at the integration boundary."""


def _request_payload(request: ExperimentRequest) -> dict[str, Any]:
    return {
        "schema": "ai-scientist-mini.experiment-request",
        "schema_version": API_VERSION,
        "request": request.to_dict(),
    }


def _result_from_payload(payload: Any, *, fallback_provider: str, experiment_id: str) -> ExperimentResult:
    """Normalize responses from a provider process or HTTP endpoint.

    Both a wrapped response (``{"result": {...}}``) and a bare result are
    accepted.  This makes the boundary pleasant for small scripts while still
    retaining a versioned envelope for production integrations.
    """

    if isinstance(payload, Mapping) and "result" in payload:
        payload = payload["result"]
    if not isinstance(payload, Mapping):
        raise AdapterError("provider response must be a JSON object")
    data = dict(payload)
    data.setdefault("experiment_id", experiment_id)
    data.setdefault("provider_id", fallback_provider)
    try:
        return ExperimentResult.from_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise AdapterError(f"invalid provider result: {exc}") from exc


class LocalProviderAdapter:
    """Wrap a Python provider object without coupling to its module path."""

    def __init__(self, provider: ExperimentProvider, *, adapter_id: str | None = None,
                 requires_approval: bool = False):
        if not callable(getattr(provider, "run", None)):
            raise TypeError("provider must implement run(request, ...)" )
        self.provider = provider
        self.provider_id = str(adapter_id or getattr(provider, "provider_id", provider.__class__.__name__))
        self.requires_approval = requires_approval
        self.descriptor = provider_to_descriptor(provider, adapter_id=self.provider_id)
        if requires_approval:
            self.descriptor = AdapterDescriptor(
                **{**self.descriptor.to_dict(), "requires_approval": True}
            )

    @property
    def capabilities(self) -> ProviderCapabilities:
        value = getattr(self.provider, "capabilities", ProviderCapabilities())
        return ProviderCapabilities.from_dict(value) if isinstance(value, Mapping) else value

    def health(self) -> Mapping[str, Any]:
        health = getattr(self.provider, "health", None)
        if callable(health):
            result = health()
            return dict(result) if isinstance(result, Mapping) else {"status": str(result)}
        return {"status": "ready", "provider_id": self.provider_id}

    def estimate_cost(self, request: ExperimentRequest) -> float:
        estimate = getattr(self.provider, "estimate_cost", None)
        if callable(estimate):
            return max(0.0, float(estimate(request)))
        return 0.0

    def run(self, request: ExperimentRequest, *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        if cancel and cancel():
            return ExperimentResult(
                experiment_id=request.experiment_id,
                provider_id=self.provider_id,
                status="cancelled",
                error_type="Cancelled",
                error_message="cancelled before provider invocation",
            )
        try:
            result = self.provider.run(request, cancel=cancel)
        except TypeError:
            # Compatibility with simple first-party providers that do not
            # expose the optional cancel keyword.
            result = self.provider.run(request)
        except Exception as exc:  # provider failures are auditable data
            return ExperimentResult(
                experiment_id=request.experiment_id,
                provider_id=self.provider_id,
                status="failed",
                error_type="ProviderError",
                error_message=str(exc),
            )
        if isinstance(result, ExperimentResult):
            result.provider_id = result.provider_id or self.provider_id
            result.experiment_id = result.experiment_id or request.experiment_id
            return result
        return _result_from_payload(result, fallback_provider=self.provider_id,
                                    experiment_id=request.experiment_id)


class LegacyProviderAdapter:
    """Bridge the built-in ``ExperimentDesign`` provider API to this boundary.

    The first-party engine historically exposes ``execute(design, seed,
    run_index)`` while external integrations use JSON ``ExperimentRequest``
    records.  This adapter keeps the two APIs separate and imports the domain
    model lazily, so the transport layer remains usable on its own.
    """

    def __init__(self, provider: Any, *, adapter_id: str | None = None,
                 design_factory: Callable[[ExperimentRequest], Any] | None = None):
        if not callable(getattr(provider, "execute", None)) and not callable(getattr(provider, "run", None)):
            raise TypeError("legacy provider must implement execute() or run()")
        self.provider = provider
        self.provider_id = str(adapter_id or getattr(provider, "name", provider.__class__.__name__))
        self.design_factory = design_factory
        self.capabilities = ProviderCapabilities(
            provider_type="experiment",
            supports_repeated_runs=True,
            supports_cancellation=False,
            external=False,
        )
        self.descriptor = AdapterDescriptor(
            adapter_id=self.provider_id,
            display_name=self.provider_id,
            version=API_VERSION,
            transport="local-domain",
            description="Adapter for the built-in ExperimentDesign provider contract",
        )

    def health(self) -> Mapping[str, Any]:
        return {"status": "ready", "provider_id": self.provider_id, "transport": "local-domain"}

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return 0.0

    def _make_design(self, request: ExperimentRequest) -> Any:
        if self.design_factory:
            return self.design_factory(request)
        # Importing the local domain model here avoids making JSON-only users
        # depend on the rest of the application at module import time.
        try:
            from ..models import ExperimentDesign
        except ImportError as exc:  # pragma: no cover - only for standalone use
            raise AdapterError("built-in domain models are unavailable") from exc
        config = dict(request.config or {})
        procedure = config.get("procedure", [])
        if isinstance(procedure, str):
            procedure = [procedure]
        return ExperimentDesign(
            id=request.experiment_id,
            hypothesis_id=request.hypothesis_id or "",
            name=str(config.get("name", config.get("strategy", request.experiment_id))),
            independent_variable=str(config.get("independent_variable", config.get("strategy", ""))),
            dependent_variable=str(config.get("dependent_variable", config.get("metric", "score"))),
            control=str(config.get("control", "")),
            dataset=str(config.get("dataset", request.dataset or "synthetic")),
            sample_size=max(1, int(config.get("sample_size", request.runs))),
            seed=int(request.seed),
            metric=str(config.get("metric", "score")),
            procedure=[str(item) for item in procedure],
            success_criteria=str(config.get("success_criteria", "")),
            config=config,
        )

    @staticmethod
    def _normalize_provider_result(value: Any, *, run_id: str, seed: int) -> RunResult:
        if isinstance(value, RunResult):
            return value
        metrics_raw = getattr(value, "metrics", None)
        raw_result = getattr(value, "raw_result", {})
        logs_raw = getattr(value, "logs", [])
        runtime = getattr(value, "runtime_seconds", None)
        error = getattr(value, "error", None)
        if isinstance(value, Mapping):
            metrics_raw = value.get("metrics", value)
            raw_result = value.get("raw_result", dict(value))
            logs_raw = value.get("logs", [])
            runtime = value.get("runtime_seconds")
            error = value.get("error")
        metrics: dict[str, float] = {}
        if isinstance(metrics_raw, Mapping):
            for key, item in metrics_raw.items():
                try:
                    metrics[str(key)] = float(item)
                except (TypeError, ValueError):
                    continue
        return RunResult(
            run_id=run_id,
            status="failed" if error else "complete",
            metrics=metrics,
            observations=dict(raw_result) if isinstance(raw_result, Mapping) else {"value": raw_result},
            seed=seed,
            duration_seconds=float(runtime) if runtime is not None else None,
            error_type="ProviderError" if error else None,
            error_message=str(error) if error else None,
            logs=[{"message": str(item)} for item in (logs_raw or [])],
        )

    def run(self, request: ExperimentRequest, *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        design = self._make_design(request)
        runs: list[RunResult] = []
        started = time.perf_counter()
        for index in range(max(1, int(request.runs))):
            if cancel and cancel():
                return ExperimentResult(request.experiment_id, self.provider_id, status="cancelled", runs=runs,
                                        public_summary="cancelled by host")
            seed = int(request.seed) + index
            run_id = f"{request.experiment_id}:run-{index + 1}"
            try:
                execute = getattr(self.provider, "execute", None) or getattr(self.provider, "run")
                value = execute(design, seed=seed, run_index=index)
                runs.append(self._normalize_provider_result(value, run_id=run_id, seed=seed))
            except Exception as exc:
                runs.append(RunResult(run_id=run_id, status="failed", seed=seed,
                                      error_type="ProviderError", error_message=str(exc)))
        metric_values: dict[str, list[float]] = {}
        for item in runs:
            if item.status == "complete":
                for key, value in item.metrics.items():
                    metric_values.setdefault(key, []).append(float(value))
        aggregate = {key: statistics.fmean(values) for key, values in metric_values.items() if values}
        failed = [item for item in runs if item.status == "failed"]
        status = "failed" if failed and len(failed) == len(runs) else "complete"
        return ExperimentResult(
            experiment_id=request.experiment_id,
            provider_id=self.provider_id,
            status=status,
            runs=runs,
            aggregate_metrics=aggregate,
            reproducibility={"seed": request.seed, "runs": request.runs},
            public_summary=f"Executed {len(runs)} local provider run(s) in {time.perf_counter() - started:.3f}s",
            error_type="ProviderError" if status == "failed" else None,
            error_message=failed[0].error_message if status == "failed" else None,
        )


class JsonLineAdapter:
    """Invoke an external provider command using one JSON request/response.

    The command receives a single UTF-8 JSON line on stdin and must return a
    single JSON object on stdout.  Stderr is retained in a failed result's
    audit metadata, never silently printed into the JSON channel.
    """

    def __init__(self, command: Sequence[str], *, adapter_id: str,
                 display_name: str | None = None, timeout_seconds: float = 300.0,
                 env: Mapping[str, str] | None = None,
                 description: str = "JSON-line experiment provider",
                 requires_approval: bool = True):
        if not command:
            raise ValueError("command cannot be empty")
        self.command = tuple(str(part) for part in command)
        self.provider_id = adapter_id
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.env = dict(env or {})
        self.descriptor = AdapterDescriptor(
            adapter_id=adapter_id,
            display_name=display_name or adapter_id,
            version=API_VERSION,
            transport="json-line",
            description=description,
            external=True,
            requires_approval=requires_approval,
            command=self.command,
        )
        self.capabilities = ProviderCapabilities(external=True, supports_cancellation=False)

    def health(self) -> Mapping[str, Any]:
        return {"status": "configured", "provider_id": self.provider_id,
                "transport": "json-line", "command": list(self.command)}

    def estimate_cost(self, request: ExperimentRequest) -> float:
        # External providers may override this through a metadata field.  A
        # missing estimate is intentionally conservative for callers: they
        # can reject it in their budget policy before run().
        value = request.metadata.get("estimated_cost", 0.0)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def run(self, request: ExperimentRequest, *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        if cancel and cancel():
            return ExperimentResult(request.experiment_id, self.provider_id, status="cancelled",
                                    error_type="Cancelled", error_message="cancelled before launch")
        env = os.environ.copy()
        env.update(self.env)
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            payload = json.dumps(_request_payload(request), ensure_ascii=False, separators=(",", ":")) + "\n"
            try:
                stdout, stderr = process.communicate(payload, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return ExperimentResult(
                    experiment_id=request.experiment_id, provider_id=self.provider_id,
                    status="failed", error_type="Timeout",
                    error_message=f"provider exceeded {self.timeout_seconds:g}s",
                    metadata={"stderr": stderr[-4000:]},
                )
        except OSError as exc:
            return ExperimentResult(
                experiment_id=request.experiment_id, provider_id=self.provider_id,
                status="failed", error_type="InfrastructureFailure", error_message=str(exc),
            )
        if cancel and cancel():
            return ExperimentResult(request.experiment_id, self.provider_id, status="cancelled",
                                    error_type="Cancelled", error_message="cancelled after provider run")
        if process.returncode != 0:
            return ExperimentResult(
                experiment_id=request.experiment_id, provider_id=self.provider_id,
                status="failed", error_type="ProviderProcessError",
                error_message=f"provider exited with code {process.returncode}",
                metadata={"stderr": stderr[-4000:]},
            )
        try:
            response = json.loads(stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            return ExperimentResult(
                experiment_id=request.experiment_id, provider_id=self.provider_id,
                status="failed", error_type="InvalidProviderResponse",
                error_message=str(exc), metadata={"stdout": stdout[-4000:], "stderr": stderr[-4000:]},
            )
        try:
            result = _result_from_payload(response, fallback_provider=self.provider_id,
                                         experiment_id=request.experiment_id)
        except AdapterError as exc:
            return ExperimentResult(
                experiment_id=request.experiment_id, provider_id=self.provider_id,
                status="failed", error_type="InvalidProviderResponse", error_message=str(exc),
                metadata={"stdout": stdout[-4000:], "stderr": stderr[-4000:]},
            )
        if stderr.strip():
            result.metadata.setdefault("provider_stderr", stderr[-4000:])
        return result


class HttpJsonAdapter:
    """Call an explicitly configured HTTP JSON endpoint.

    This adapter is intentionally tiny and uses the standard library.  It is
    disabled by default in the application; callers should require human
    approval and apply their own network allow-list before invoking it.
    """

    def __init__(self, endpoint: str, *, adapter_id: str,
                 display_name: str | None = None, timeout_seconds: float = 60.0,
                 headers: Mapping[str, str] | None = None,
                 requires_approval: bool = True):
        if not endpoint.lower().startswith(("http://", "https://")):
            raise ValueError("endpoint must use http:// or https://")
        self.endpoint = endpoint
        self.provider_id = adapter_id
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}
        self.descriptor = AdapterDescriptor(
            adapter_id=adapter_id,
            display_name=display_name or adapter_id,
            version=API_VERSION,
            transport="http-json",
            description="HTTP JSON experiment provider",
            external=True,
            requires_approval=requires_approval,
            endpoint=endpoint,
        )
        self.capabilities = ProviderCapabilities(external=True, supports_cancellation=False)

    def health(self) -> Mapping[str, Any]:
        return {"status": "configured", "provider_id": self.provider_id,
                "transport": "http-json", "endpoint": self.endpoint}

    def estimate_cost(self, request: ExperimentRequest) -> float:
        try:
            return max(0.0, float(request.metadata.get("estimated_cost", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def run(self, request: ExperimentRequest, *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        if cancel and cancel():
            return ExperimentResult(request.experiment_id, self.provider_id, status="cancelled",
                                    error_type="Cancelled", error_message="cancelled before request")
        body = json.dumps(_request_payload(request), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(self.endpoint, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-4000:]
            return ExperimentResult(request.experiment_id, self.provider_id, status="failed",
                                    error_type="ProviderHTTPError",
                                    error_message=f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ExperimentResult(request.experiment_id, self.provider_id, status="failed",
                                    error_type="InfrastructureFailure", error_message=str(exc))
        try:
            response_payload = json.loads(raw)
            return _result_from_payload(response_payload, fallback_provider=self.provider_id,
                                        experiment_id=request.experiment_id)
        except (json.JSONDecodeError, AdapterError) as exc:
            return ExperimentResult(request.experiment_id, self.provider_id, status="failed",
                                    error_type="InvalidProviderResponse", error_message=str(exc),
                                    metadata={"response": raw[-4000:]})


class AdapterRegistry:
    """In-process registry for adapters exposed to the dashboard/API/CLI."""

    def __init__(self):
        self._adapters: dict[str, Any] = {}
        self._lock = threading.RLock()

    def register(self, adapter: Any, *, replace: bool = False) -> AdapterDescriptor:
        descriptor = getattr(adapter, "descriptor", None)
        if descriptor is None:
            descriptor = provider_to_descriptor(adapter)
        if isinstance(descriptor, Mapping):
            descriptor = AdapterDescriptor.from_dict(descriptor)
        adapter_id = str(descriptor.adapter_id)
        with self._lock:
            if adapter_id in self._adapters and not replace:
                raise ValueError(f"adapter already registered: {adapter_id}")
            self._adapters[adapter_id] = adapter
        return descriptor

    def unregister(self, adapter_id: str) -> bool:
        with self._lock:
            return self._adapters.pop(adapter_id, None) is not None

    def get(self, adapter_id: str) -> Any:
        with self._lock:
            try:
                return self._adapters[adapter_id]
            except KeyError as exc:
                raise KeyError(f"unknown adapter: {adapter_id}") from exc

    def list_descriptors(self) -> list[AdapterDescriptor]:
        with self._lock:
            descriptors: list[AdapterDescriptor] = []
            for adapter in self._adapters.values():
                descriptor = getattr(adapter, "descriptor", None)
                if isinstance(descriptor, Mapping):
                    descriptor = AdapterDescriptor.from_dict(descriptor)
                if descriptor is None:
                    descriptor = provider_to_descriptor(adapter)
                descriptors.append(descriptor)
            return descriptors

    def describe(self) -> list[dict[str, Any]]:
        return [descriptor.to_dict() for descriptor in self.list_descriptors()]

    def health(self, adapter_id: str | None = None) -> dict[str, Any]:
        if adapter_id:
            adapters = [(adapter_id, self.get(adapter_id))]
        else:
            with self._lock:
                adapters = list(self._adapters.items())
        output: dict[str, Any] = {}
        for key, adapter in adapters:
            try:
                value = adapter.health() if callable(getattr(adapter, "health", None)) else {"status": "ready"}
                output[key] = dict(value) if isinstance(value, Mapping) else {"status": str(value)}
            except Exception as exc:
                output[key] = {"status": "error", "error": str(exc)}
        return output

    def run(self, adapter_id: str, request: ExperimentRequest,
            *, cancel: Callable[[], bool] | None = None) -> ExperimentResult:
        adapter = self.get(adapter_id)
        if not callable(getattr(adapter, "run", None)):
            raise AdapterError(f"adapter {adapter_id!r} has no run method")
        return adapter.run(request, cancel=cancel)

    def estimate_cost(self, adapter_id: str, request: ExperimentRequest) -> float:
        adapter = self.get(adapter_id)
        estimate = getattr(adapter, "estimate_cost", None)
        if not callable(estimate):
            return 0.0
        return max(0.0, float(estimate(request)))

    def register_manifest(self, manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, replace: bool = False) -> list[AdapterDescriptor]:
        """Register adapters described by a JSON manifest.

        A manifest contains only transport metadata, for example::

            {"adapters": [{"adapter_id": "arena", "transport": "json-line",
                            "command": ["arena-worker"]}]}

        Local Python objects are intentionally not importable from a manifest;
        this keeps the integration boundary explicit and avoids executing
        arbitrary module paths from untrusted JSON.
        """

        entries: Any = manifest.get("adapters", []) if isinstance(manifest, Mapping) else manifest
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("manifest.adapters must be a list")
        descriptors: list[AdapterDescriptor] = []
        for raw in entries:
            descriptor = AdapterDescriptor.from_dict(raw)
            if descriptor.transport == "json-line":
                if not descriptor.command:
                    raise ValueError(f"json-line adapter {descriptor.adapter_id!r} has no command")
                adapter = JsonLineAdapter(
                    descriptor.command,
                    adapter_id=descriptor.adapter_id,
                    display_name=descriptor.display_name,
                    description=descriptor.description,
                    requires_approval=descriptor.requires_approval,
                )
            elif descriptor.transport == "http-json":
                if not descriptor.endpoint:
                    raise ValueError(f"http-json adapter {descriptor.adapter_id!r} has no endpoint")
                adapter = HttpJsonAdapter(
                    descriptor.endpoint,
                    adapter_id=descriptor.adapter_id,
                    display_name=descriptor.display_name,
                    requires_approval=descriptor.requires_approval,
                )
            else:
                raise ValueError(
                    f"manifest transport {descriptor.transport!r} is not executable; "
                    "register local providers in Python"
                )
            descriptors.append(self.register(adapter, replace=replace))
        return descriptors

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe adapter manifest for persistence or inspection."""

        return {"schema": "ai-scientist-mini.adapter-manifest", "schema_version": API_VERSION,
                "adapters": self.describe()}


def load_adapter_manifest(path: str | os.PathLike[str], registry: AdapterRegistry | None = None,
                          *, replace: bool = False) -> AdapterRegistry:
    """Load a transport-only adapter manifest from disk."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    target = registry or AdapterRegistry()
    target.register_manifest(value, replace=replace)
    return target
