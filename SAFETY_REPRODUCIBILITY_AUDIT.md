# Safety, Reproducibility and Scientific Audit

AI Scientist Mini 的目标是让实验过程可检查，而不是让软件代替科学判断。本文件是
开发和验收时的最低要求。

## 1. Safety defaults

| Area | Default | Required override |
|---|---|---|
| Provider | Synthetic / Mock | User explicitly selects another adapter |
| Network | Disabled | Explicit adapter configuration and approval |
| API key | Not read | Environment/secret reference supplied by user |
| Estimated cost | `0` | Provider reports a non-zero estimate |
| Max experiments/runs | Finite project values | User edits budget before execution |
| Runtime | Finite per-run and project limits | User explicitly raises limit |
| Human approval | Optional; recommended for external providers | Explicit project setting |
| Publishing | Never automatic | User exports and publishes manually |

The application must not purchase services, send results to the internet, alter unrelated
projects, or run indefinitely. The stop button and process cancellation path must be visible in
the UI and exposed through the service layer.

## 2. Resource-control protocol

Before an experiment is enqueued, the system validates:

- the project is active and within its maximum experiment count;
- the requested number of runs fits the remaining run budget;
- the estimated runtime and cost fit the remaining budget;
- the provider is allowed by the project mode;
- the canonical deduplication key is not already complete;
- the design has a bounded sample size and timeout.

Before each individual run, reserve the budget atomically. Release unused reservations when a
run fails or is cancelled. Persist both reservation and consumption events so a restart cannot
silently reset limits.

## 3. Failure taxonomy

Every failed execution must include one category and a public explanation:

| Category | Meaning | Hypothesis update |
|---|---|---|
| `infrastructure_failure` | Provider/process/filesystem/timeout issue | No rejection; retry only with a reason |
| `invalid_design` | Input violates design contract | No scientific evidence |
| `insufficient_data` | Valid run, but sample/power is inadequate | Inconclusive; may schedule bounded follow-up |
| `negative_result` | Valid data contradicts a prediction | Evidence may lower confidence |

An `Experiment Failed` state is never converted directly into `Hypothesis Rejected`.

## 4. Reproducibility checklist

Each completed or failed run retains:

- canonical experiment configuration;
- seed and deterministic random-stream information;
- provider name and adapter version;
- Python/runtime/platform information;
- code/package version and optional git revision;
- dataset identifier and content hash;
- raw result, aggregate result, and result hash;
- stdout/stderr or structured logs and log hash;
- start/end time, elapsed time and cancellation reason;
- parent hypothesis/design/experiment IDs.

The generated `reproducibility_manifest.json` is immutable. A rerun receives a new `run_id` and
links to the prior manifest; historical records are never overwritten.

## 5. Scientific audit trail

The audit trail is append-only. At minimum, record these events:

```text
project.created
hypothesis.created
design.generated
experiment.queued
experiment.deduplicated
budget.reserved
run.started
run.completed | run.failed | run.cancelled
analysis.completed
evidence.created
hypothesis.updated
experiment.selected
report.generated
```

Every event contains:

```json
{
  "event_id": "evt-...",
  "timestamp": "RFC-3339",
  "actor": "system|user|rule|provider",
  "action": "hypothesis.updated",
  "entity_type": "hypothesis",
  "entity_id": "hyp-...",
  "inputs": ["result-...", "evidence-..."],
  "rule_id": "confidence.update.v1",
  "before": {"confidence": 0.40, "status": "Untested"},
  "after": {"confidence": 0.58, "status": "Inconclusive"},
  "reason_summary": "Observed effect was positive but uncertainty crossed the inconclusive threshold",
  "content_hash": "sha256:..."
}
```

The public journal is a filtered projection of audit events. It must not contain hidden model
reasoning, credentials, raw secrets, or private chain-of-thought.

## 6. Transparent update rules

An update rule must document:

1. required evidence fields;
2. sample-size and uncertainty thresholds;
3. confidence delta bounds;
4. status-transition conditions;
5. rule ID and version;
6. an example showing the same input produces the same output.

The report should say “supported under the tested synthetic setup” instead of “scientifically
proven”. If evidence is missing, stale, duplicated or based on failed runs, the rule must leave
the hypothesis unchanged or mark it inconclusive.

## 7. Human Approval mode

`Autonomous Sandbox` may run local Mock/Synthetic experiments automatically within the budget.
`Human Approval` pauses before an external provider, network access, paid API call, or any action
whose estimated cost is non-zero. An approval record includes the requested action, provider,
estimated cost/runtime, scope and approver timestamp. Denial leaves the experiment `Cancelled`
with no provider call.

## 8. Security and privacy

- Do not put API keys in source, JSON configs, screenshots, reports or audit events.
- Redact environment variables and file paths containing secrets before report export.
- Python function providers use an explicit registry; do not execute arbitrary code from a
  project JSON file.
- Network-capable adapters are disabled in the default build and should declare permissions.
- Exported reports contain only user-selected public data and synthetic results by default.

## 9. Release gate

A build is releasable only if all of these pass:

- no-network test suite succeeds;
- budget limits stop both serial and concurrent runs;
- stop button cancels a running provider cooperatively;
- duplicate designs are not executed twice unintentionally;
- restart recovery preserves queue, budget, evidence and audit records;
- every conclusion in the demo report links to hypothesis, experiment, run and result IDs;
- report generation is deterministic for a fixed database and seed;
- no secret or chain-of-thought content appears in exported artifacts.
