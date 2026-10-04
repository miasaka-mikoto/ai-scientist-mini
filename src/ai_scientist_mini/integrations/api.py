"""Small local HTTP/JSON API for tool and future-project integration.

This is a boundary, not the application's dashboard server.  It intentionally
uses only the Python standard library and defaults to loopback-only operation.
External adapters are denied unless the caller explicitly opts in.
"""

from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .adapters import AdapterError, AdapterRegistry
from .contracts import ExperimentRequest
from .exchange import ExchangeError, make_envelope, parse_envelope


class IntegrationService:
    """Request/response logic independent of an HTTP server.

    ``run_guard`` is called immediately before execution.  The host engine can
    use it to enforce max experiments, max runs, estimated cost, and human
    approval.  Returning ``False`` produces a blocked result rather than
    silently executing an experiment.
    """

    def __init__(self, registry: AdapterRegistry | None = None, *, allow_external: bool = False,
                 run_guard: Callable[[str, ExperimentRequest], bool | tuple[bool, str]] | None = None):
        self.registry = registry or AdapterRegistry()
        self.allow_external = bool(allow_external)
        self.run_guard = run_guard
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {"status": "ready", "api_version": "1.0", "allow_external": self.allow_external,
                "adapters": self.registry.health()}

    def adapters(self) -> list[dict[str, Any]]:
        return self.registry.describe()

    @staticmethod
    def _request_from_payload(payload: Mapping[str, Any]) -> ExperimentRequest:
        if "request" in payload and isinstance(payload["request"], Mapping):
            payload = payload["request"]
        if "payload" in payload and isinstance(payload["payload"], Mapping):
            payload = payload["payload"]
        return ExperimentRequest.from_dict(payload)

    def validate_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            request = self._request_from_payload(payload)
        except (TypeError, ValueError, KeyError) as exc:
            return {"valid": False, "errors": [str(exc)]}
        errors: list[str] = []
        if not request.experiment_id.strip():
            errors.append("experiment_id is required")
        if request.runs < 1:
            errors.append("runs must be at least 1")
        if request.seed < 0:
            errors.append("seed must be non-negative")
        return {"valid": not errors, "errors": errors, "request": request.to_dict()}

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        adapter_id = str(payload.get("adapter_id", ""))
        if not adapter_id:
            raise AdapterError("adapter_id is required")
        request = self._request_from_payload(payload)
        descriptor = next((d for d in self.registry.list_descriptors() if d.adapter_id == adapter_id), None)
        if descriptor is None:
            raise AdapterError(f"unknown adapter: {adapter_id}")
        if descriptor.external and not self.allow_external:
            return {
                "accepted": False,
                "status": "blocked",
                "reason": "external adapters are disabled; enable explicitly and require approval",
                "adapter_id": adapter_id,
                "experiment_id": request.experiment_id,
            }
        if self.run_guard:
            decision = self.run_guard(adapter_id, request)
            allowed, reason = (decision if isinstance(decision, tuple) else (bool(decision), "blocked by run guard"))
            if not allowed:
                return {"accepted": False, "status": "blocked", "reason": reason,
                        "adapter_id": adapter_id, "experiment_id": request.experiment_id}
        token = uuid.uuid4().hex
        cancel_event = threading.Event()
        with self._lock:
            self._cancel[token] = cancel_event
        try:
            result = self.registry.run(adapter_id, request, cancel=cancel_event.is_set)
            return {"accepted": True, "execution_id": token, "result": result.to_dict()}
        finally:
            with self._lock:
                self._cancel.pop(token, None)

    def cancel(self, execution_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(execution_id)
        if event is None:
            return False
        event.set()
        return True


class _Handler(BaseHTTPRequestHandler):
    server: "IntegrationHTTPServer"

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Mapping[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size)
            value = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExchangeError(f"invalid JSON body: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ExchangeError("JSON body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path.rstrip("/") or "/"
        service = self.server.service
        if route == "/health":
            self._json(HTTPStatus.OK, service.health())
        elif route == "/adapters":
            self._json(HTTPStatus.OK, make_envelope("adapters", service.adapters()))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path.rstrip("/") or "/"
        service = self.server.service
        try:
            payload = self._body()
            if route == "/validate-request":
                self._json(HTTPStatus.OK, make_envelope("validation", service.validate_request(payload)))
            elif route == "/run":
                self._json(HTTPStatus.OK, make_envelope("experiment-result", service.run(payload)))
            elif route == "/cancel":
                execution_id = str(payload.get("execution_id", ""))
                self._json(HTTPStatus.OK, {"cancelled": service.cancel(execution_id), "execution_id": execution_id})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (AdapterError, ExchangeError, KeyError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # keep server alive for malformed plugin code
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        # Integrations should not write noisy access logs to stdout.  The host
        # may wrap the server with its own audit logger.
        return


class IntegrationHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: IntegrationService):
        self.service = service
        super().__init__(address, _Handler)


def create_api_server(service: IntegrationService | None = None, *, host: str = "127.0.0.1",
                      port: int = 8765) -> IntegrationHTTPServer:
    """Create (but do not start) a loopback API server."""

    return IntegrationHTTPServer((host, int(port)), service or IntegrationService())


def serve_api(service: IntegrationService | None = None, *, host: str = "127.0.0.1",
              port: int = 8765) -> None:
    server = create_api_server(service, host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()

