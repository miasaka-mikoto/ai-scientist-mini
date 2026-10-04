from __future__ import annotations

import json

from aiscientist.ui import EngineAdapter, LocalDemoEngine, headless_snapshot


def test_headless_demo_completes_two_rounds_without_external_api():
    state = headless_snapshot(LocalDemoEngine(), run_demo=True)
    assert len(state["hypotheses"]) == 5
    assert len(state["experiments"]) >= 6
    assert len(state["runs"]) >= 20
    assert state["project"]["budget"]["cost_used"] == 0.0
    assert state["best_hypothesis"]["id"]
    json.dumps(state, ensure_ascii=False)


def test_adapter_accepts_plain_snapshot_engine():
    class Stub:
        def snapshot(self):
            return {"project": {"research_question": "q"}, "hypotheses": []}

    state = EngineAdapter(Stub()).snapshot()
    assert state["project"]["research_question"] == "q"

