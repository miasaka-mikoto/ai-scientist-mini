from pathlib import Path

from ai_scientist_mini import ScientistEngine
from ai_scientist_mini.models import ResearchBudget
from ai_scientist_mini.persistence import SQLiteProjectStore, load_project, save_sqlite


def test_sqlite_checkpoint_round_trip_and_audit_index(tmp_path: Path):
    engine = ScientistEngine()
    project = engine.create_project(
        "Which memory strategy is best for a long-term agent?",
        budget=ResearchBudget(maximum_experiments=8, maximum_runs=32, estimated_api_cost=0.0),
    )
    engine.propose_hypotheses()
    db = tmp_path / "study.sqlite3"
    save_sqlite(project, db)
    restored = SQLiteProjectStore().load(db)
    assert restored.id == project.id
    assert len(restored.hypotheses) == 5
    assert SQLiteProjectStore().audit_rows(db)
    assert load_project(db).id == project.id


def test_core_cycle_runs_two_selection_rounds_with_fresh_seed(tmp_path: Path):
    engine = ScientistEngine()
    project = engine.create_project(
        "Which memory strategy is best for a long-term agent?",
        budget=ResearchBudget(maximum_experiments=10, maximum_runs=60, estimated_api_cost=0.0),
    )
    completed = engine.run_scientific_cycle(rounds=2, runs_per_experiment=4)
    assert len(project.hypotheses) >= 5
    assert len(project.experiments) >= 7
    assert sum(len(item.runs) for item in project.experiments) >= 20
    assert len({item.design.seed for item in project.experiments}) == len(project.experiments)
    assert completed
    state = tmp_path / "state.json"
    engine.save_state(state)
    resumed = ScientistEngine.load_state(state)
    assert resumed.project.id == project.id
    assert len(resumed.project.audit_trail) >= len(project.audit_trail)
