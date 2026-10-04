"""CLI for the versioned integration boundary.

The main application may expose a richer command line, but these commands are
stable enough for future projects and scripts to use independently:

``providers``  list registered adapters
``validate``   validate a request JSON file
``run``        run a request through a registered adapter
``export``     wrap arbitrary state in a versioned JSON envelope
``import``     inspect an envelope
``api``        start the local loopback API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .adapters import AdapterRegistry, HttpJsonAdapter, JsonLineAdapter
from .api import IntegrationService, serve_api
from .contracts import ExperimentRequest
from .exchange import ExchangeError, export_json, import_envelope, import_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-scientist-integrations",
                                     description="AI Scientist Mini integration boundary")
    sub = parser.add_subparsers(dest="command", required=True)

    providers = sub.add_parser("providers", help="list registered adapters")
    providers.add_argument("--json", action="store_true", dest="as_json")

    validate = sub.add_parser("validate", help="validate an experiment request JSON file")
    validate.add_argument("input", type=Path)

    run = sub.add_parser("run", help="run an experiment request through an adapter")
    run.add_argument("adapter_id")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--allow-external", action="store_true",
                     help="explicitly allow an external adapter")
    run.add_argument("--json-line-command", nargs="+", metavar="COMMAND",
                     help="register adapter as a JSON-line command (for standalone use)")
    run.add_argument("--http-endpoint", metavar="URL",
                     help="register adapter as an HTTP JSON endpoint (for standalone use)")

    export = sub.add_parser("export", help="wrap a JSON file in a versioned envelope")
    export.add_argument("input", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--kind", default="record")

    inspect = sub.add_parser("import", help="inspect a versioned JSON envelope")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--kind")
    inspect.add_argument("--payload-only", action="store_true")

    api = sub.add_parser("api", help="serve the local integration API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8765)
    api.add_argument("--allow-external", action="store_true")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None, *, registry: AdapterRegistry | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = registry or AdapterRegistry()
    if args.command == "providers":
        value = registry.describe()
        _print_json(value) if args.as_json else [print(f"{item['adapter_id']}\t{item['display_name']}\t{item['transport']}") for item in value]
        return 0

    if args.command == "validate":
        try:
            payload = import_json(args.input, require_envelope=False)
            service = IntegrationService(registry)
            _print_json(service.validate_request(payload))
            return 0 if service.validate_request(payload).get("valid") else 2
        except (OSError, ExchangeError, ValueError) as exc:
            print(f"validate error: {exc}", file=sys.stderr)
            return 2

    if args.command == "export":
        try:
            payload = import_json(args.input, require_envelope=False)
            export_json(payload, args.output, kind=args.kind)
            print(str(args.output))
            return 0
        except (OSError, ExchangeError, ValueError) as exc:
            print(f"export error: {exc}", file=sys.stderr)
            return 2

    if args.command == "import":
        try:
            value = import_json(args.input, expected_kind=args.kind,
                               require_envelope=not args.payload_only)
            _print_json(value)
            return 0
        except (OSError, ExchangeError, ValueError) as exc:
            print(f"import error: {exc}", file=sys.stderr)
            return 2

    if args.command == "run":
        if args.json_line_command:
            registry.register(JsonLineAdapter(args.json_line_command, adapter_id=args.adapter_id), replace=True)
        elif args.http_endpoint:
            registry.register(HttpJsonAdapter(args.http_endpoint, adapter_id=args.adapter_id), replace=True)
        try:
            payload = import_json(args.input, require_envelope=False)
            request_payload = payload.get("payload", payload) if isinstance(payload, dict) else payload
            request = ExperimentRequest.from_dict(request_payload.get("request", request_payload)
                                                  if isinstance(request_payload, dict) else {})
            service = IntegrationService(registry, allow_external=args.allow_external)
            result = service.run({"adapter_id": args.adapter_id, "request": request.to_dict()})
            if args.output:
                export_json(result, args.output, kind="execution")
            _print_json(result)
            return 0 if result.get("accepted") and result.get("result", {}).get("status") != "failed" else 2
        except (OSError, ExchangeError, ValueError, KeyError) as exc:
            print(f"run error: {exc}", file=sys.stderr)
            return 2

    if args.command == "api":
        service = IntegrationService(registry, allow_external=args.allow_external)
        serve_api(service, host=args.host, port=args.port)
        return 0

    return 2

