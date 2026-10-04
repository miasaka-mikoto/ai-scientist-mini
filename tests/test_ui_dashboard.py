"""Headless tests for the AI Scientist Mini dashboard contract."""

from ai_scientist_mini.ui.app import run_smoke_test
from ai_scientist_mini.ui.dashboard import DashboardController, DashboardViewModel, LocalDemoService


def test_local_demo_completes_scientific_loop():
    result = run_smoke_test()
    assert result["passed"] is True
    assert result["experiments"] >= 5
    assert result["evidence"] >= 5


def test_view_model_accepts_engine_style_mapping():
    vm = DashboardViewModel.from_snapshot(
        {
            "research_question": "Does A beat B?",
            "status": "Ready",
            "hypotheses": [{"id": "H1", "statement": "A wins", "confidence": 0.75}],
            "budget": {"used_experiments": 1, "max_experiments": 4, "estimated_api_cost": 0},
        }
    )
    assert vm.question == "Does A beat B?"
    assert vm.confidence == 0.75
    assert "1/4" in vm.budget_text


def test_controller_supports_stop_and_resume_without_tk():
    service = LocalDemoService()
    controller = DashboardController(service)
    controller.start_demo()
    controller.run_next()
    stopped = controller.stop()
    assert stopped.status == "Stopped"
    resumed = controller.resume()
    assert resumed.question

