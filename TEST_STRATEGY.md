# Test Strategy and Acceptance Matrix

测试必须在无网络、无 API key、无外部项目源码的干净环境中可运行。任何需要真实
Provider 的测试都不是默认验收的一部分。

## 1. Test layers

### Unit tests

覆盖纯函数和小组件：

- canonical config / design hash / dedupe key；
- seed handling and deterministic SyntheticProvider output；
- mean, median, standard deviation, confidence interval, effect size, success rate；
- hypothesis status/confidence update rules；
- selection heuristics and tie-breaking；
- budget reservation/release and failure classification；
- report/audit serialization and redaction。

### Integration tests

- create project → generate hypotheses → design → queue → run Mock/Synthetic → analyze → update;
- two selection rounds with persistent state;
- duplicate design is skipped and linked to prior result;
- failed execution does not reject a hypothesis;
- stop button cancels a running provider;
- restart recovers queued experiments and stale running records safely;
- JSON export/import preserves IDs, hashes and lineage.

### End-to-end smoke test

Run the Memory Strategy Study from an empty temporary directory. Assert:

```text
≥ 5 hypotheses
≥ 5 experiment configurations
≥ 20 completed runs
2 selection rounds
≥ 1 chart
1 complete report
all conclusions have audit lineage
estimated API cost == 0
network calls == 0
```

## 2. Acceptance matrix

| ID | Requirement | Verification | Expected |
|---|---|---|---|
| A01 | Launch | CLI/UI smoke | Application starts without network |
| A02 | Create question | integration test | Project persisted with budget |
| A03 | Candidate hypotheses | service test | Five Memory strategies created |
| A04 | Experiment plans | designer test | Required fields and success criteria present |
| A05 | Mock execution | provider test | Runs complete deterministically |
| A06 | Queue states | state-machine test | Queued/Running/Complete/Failed/Cancelled valid |
| A07 | Statistics | analyzer test | Mean, median, std, CI, effect size, success rate |
| A08 | Hypothesis update | rule test | Evidence/rule IDs recorded; no phantom conclusion |
| A09 | Second selection | e2e test | Strategy is auditable and bounded |
| A10 | Charts/report | artifact test | Figures and all required report sections |
| A11 | Restart | persistence test | State and audit trail continue after reopen |
| A12 | Deduplication | integration test | Same config+seed not rerun unnecessarily |
| A13 | Budget | safety test | Limits stop execution and emit event |
| A14 | Stop button | cancellation test | Running provider exits cooperatively |
| A15 | No paid API | offline test | No network/API key needed; cost 0 |

## 3. Determinism checks

For a fixed provider version, config, seed and dataset hash:

1. run the experiment twice in separate temporary directories;
2. compare canonical result JSON and aggregate statistics;
3. permit differences only in runtime/timestamps/opaque IDs;
4. compare report data extracts and figure source IDs.

Set `PYTHONHASHSEED=0` in CI where practical. Avoid relying on dictionary iteration order for
scientific decisions; sort all IDs before selecting or serializing.

## 4. Failure and edge cases

At minimum test:

- zero sample size, negative sample size and invalid metric;
- timeout before first sample and during a long run;
- provider exception, malformed provider result and partial artifact;
- all runs failed, some runs failed, insufficient sample;
- budget exactly exhausted and one unit over budget;
- duplicate seed vs different seed;
- concurrent reservation race;
- corrupted/old schema and stale `Running` record after restart;
- empty report and report with rejected/inconclusive hypotheses;
- user cancellation before queue and during execution.

## 5. Offline and safety tests

Tests should monkeypatch socket/network entry points and assert no outbound call occurs in the
default demo. Set fake API key values and confirm they never appear in logs/reports. Verify a
Human Approval denial results in `Cancelled` without invoking the provider.

## 6. Suggested commands

```bash
python -m pytest -q
python -m pytest -m "not slow and not external" -q
python -m pytest tests/test_e2e_demo.py -q
```

On Windows packaging, run:

```powershell
.\scripts\build_exe.ps1 -Clean -RunTests -SmokeDemo
```

The packaging smoke test should launch the produced executable with a temporary `--data-dir`,
run the demo, verify the report, then delete only that temporary directory.

## 7. Definition of done for a module

A module is complete only after:

- implementation and focused tests exist;
- error paths are classified and logged;
- persistence/restart behavior is covered if stateful;
- docs list public inputs/outputs and limitations;
- no network or paid API dependency is introduced;
- the full offline test suite remains green.
