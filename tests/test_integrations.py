import json

from ai_scientist_mini.integrations.adapters import AdapterRegistry, LegacyProviderAdapter
from ai_scientist_mini.integrations.api import IntegrationService
from ai_scientist_mini.integrations.contracts import ExperimentRequest
from ai_scientist_mini.integrations.exchange import make_envelope, parse_envelope
from ai_scientist_mini.providers import SyntheticProvider


def test_legacy_provider_adapter_round_trip_and_public_redaction():
    registry = AdapterRegistry()
    registry.register(LegacyProviderAdapter(SyntheticProvider(), adapter_id="synthetic-local"))
    service = IntegrationService(registry)
    request = ExperimentRequest(
        experiment_id="integration-exp",
        config={"strategy": "Vector", "metric": "score", "sample_size": 1},
        seed=10,
        runs=2,
    )
    response = service.run({"adapter_id": "synthetic-local", "request": request.to_dict()})
    assert response["accepted"] is True
    assert response["result"]["status"] == "complete"
    assert len(response["result"]["runs"]) == 2

    envelope = make_envelope("public", {"chain_of_thought": "must not escape", "answer": 1})
    payload = parse_envelope(envelope, expected_kind="public")["payload"]
    assert payload["chain_of_thought"] == "[redacted: private reasoning]"
    assert payload["answer"] == 1
    json.dumps(envelope, ensure_ascii=False)


def test_external_adapter_is_blocked_by_default():
    # A manifest can describe an external transport without executing it.
    registry = AdapterRegistry()
    registry.register_manifest(
        {
            "adapters": [
                {
                    "adapter_id": "future-worker",
                    "display_name": "Future worker",
                    "transport": "json-line",
                    "command": ["does-not-run-in-tests"],
                    "external": True,
                    "requires_approval": True,
                }
            ]
        }
    )
    service = IntegrationService(registry)
    result = service.run(
        {
            "adapter_id": "future-worker",
            "request": ExperimentRequest(experiment_id="blocked", runs=1).to_dict(),
        }
    )
    assert result["accepted"] is False
    assert result["status"] == "blocked"
