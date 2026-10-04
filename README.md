# AI Scientist Mini

**自动实验科学家** — a local, auditable research-loop workbench.

AI Scientist Mini turns a research question into a bounded sequence of
hypotheses, experiment configurations, synthetic/mock runs, transparent
analysis, hypothesis updates, and a reproducible report. The first release is
deliberately offline: no paid model API is called and the demo uses a
Rule-Based Scientist plus deterministic synthetic experiments.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python run_demo.py --output demo_output
python launcher.py                 # opens the desktop dashboard
python launcher.py --demo          # runs the full offline study
```

The demo creates a persistent SQLite/JSON-compatible study workspace, at least
five memory-strategy hypotheses, 20+ reproducible runs, two rounds of
experiment selection, charts, an audit trail, and a Markdown/HTML report.

For a headless run:

```bash
python -m aiscientist.cli demo --output demo_output
python -m aiscientist.cli inspect demo_output
# JSON adapter boundary (offline synthetic provider)
printf '%s\\n' '{"provider":"synthetic","seed":7,"config":{"independent_variable":"Summary","metric":"score","sample_size":8}}' | python -m aiscientist.cli adapter-run
```

The GUI is a standard-library Tkinter application. It never starts an
unbounded loop: all execution goes through an explicit budget and stop token.

`Budget.estimated_api_cost` is treated as a hard allowance (the default is
`0.0`), alongside maximum experiments, runs, and runtime.  Runtime and cost
usage are persisted and audited; a ceiling pauses the queue.

## Safety and scientific boundaries

* `Cost = 0` for all built-in providers.
* Maximum experiments, runs, runtime, and estimated API cost are enforced.
* External providers are adapters only; they are not enabled by default.
* A failed experiment is classified as infrastructure failure, invalid design,
  insufficient data, or negative result. A failure is not silently converted
  into hypothesis rejection.
* Conclusions point back to hypothesis → experiment → run → result → rule.
* The journal stores public decision summaries, never private chain-of-thought.

## Deliverables

* `src/aiscientist/`: core engine, providers, analysis, persistence, UI,
  reporting and visualization.
* `demo_output/`: generated study database, report, figures and audit log.
* `screenshots/`: reproducibly rendered PNG dashboard/chart captures
  (`scripts/render_screenshots.py`).
* `tests/`: unit and end-to-end acceptance tests.
* `ARCHITECTURE.md`: module boundaries and extension points.
* `DELIVERY_MANIFEST.json`: release file hashes and acceptance counts.
* `build_windows.ps1` / `build_linux.sh`: reproducible packaging commands.

`aiscientist.adapters.LocalJsonAdapter` is the integration seam for future
Paper2Lab, LLM Lab, Agent Arena, and Benchmark Factory connectors.  The
reserved external adapter classes intentionally raise `NotImplementedError`
until a separately reviewed integration is added; the development registry
contains only deterministic local providers.

The current build host can produce `dist/AIScientistMini` (Linux). Run
`build_windows.ps1 -Clean` on Windows, or trigger the included GitHub Actions
workflow, to produce the requested `AIScientistMini.exe`; the workflow also
runs the headless 5-hypothesis/20+ run smoke test before uploading it.

## Extension adapters

Future integrations (LLM Lab, Agent Arena, Paper2Lab and Synthetic Benchmark
Factory) should implement `ExperimentProvider` or use the JSON/CLI adapter
in `src/aiscientist/adapters.py`. The current project has no source dependency
on those projects.
