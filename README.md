# AI Scientist Mini

**自动实验科学家** — 一个本地、可复现、预算受控的规则驱动研究工作台。

AI Scientist Mini 把研究问题变成一条可检查的科学循环：

```text
Research Question → Background → Hypotheses → Experiment Design
        → Execution → Results → Analysis → Hypothesis Update
        → Next Experiment → Research Report
```

第一版完全使用 `RuleBasedScientist`、`MockLLMExperimentProvider` 和
`SyntheticProvider`。默认不会连接互联网、不会调用收费模型、不会改变其他项目，
也不会自行发布结果。真实 Provider 只能通过明确的 Adapter 接口接入。

## 当前能力

- 创建和持久化 Research Project、Hypothesis、Experiment、Run、Result、Evidence、Journal。
- 规则透明的假设生成、实验设计、结果分析和置信度更新。
- Synthetic、Python function、Mock LLM 三种本地实验后端。
- `Random`、`Best Expected Information`、`Uncertainty First`、`Follow-up Failed Result` 四种实验选择策略。
- 队列状态：`Queued`、`Running`、`Complete`、`Failed`、`Cancelled`。
- 去重检查（配置摘要 + seed + 数据哈希）和预算检查（实验、run、运行时、估算成本）。
- 研究记忆、科学审计轨迹、失败分类和可复现快照。
- Memory Strategy Study 演示：至少 5 个候选策略、5 个配置、20+ runs、两轮选择。
- 本地 Dashboard、Hypothesis Graph、图表和 Markdown/JSON 研究报告。

## 安全边界

默认配置为本地 Mock/Synthetic 模式，`estimated_api_cost=0`。系统在执行前检查：

1. 最大实验数（`max_experiments`）；
2. 最大 run 数（`max_runs`）；
3. 最大运行时间（`max_runtime_seconds`）；
4. 估算 API 成本（`estimated_api_cost` / `budget`）；
5. 停止信号和 Human Approval 模式。

失败的实验不会直接拒绝假设。失败必须标记为 `infrastructure_failure`、
`invalid_design`、`insufficient_data` 或 `negative_result`，再由透明规则决定是否更新假设。
系统只记录公开决策摘要，不要求或保存模型私有 chain-of-thought。

## 快速开始（源码）

需要 Python 3.11+。建议在 Windows PowerShell 中使用 `.venv`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ai_scientist_mini --help
```

在无网或受限构建环境中，可加 `--no-build-isolation`，避免构建阶段尝试下载依赖：

```bash
python -m pip install --no-build-isolation -e ".[dev]"
```

Linux/macOS 可使用：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ai_scientist_mini --help
```

如果当前版本提供命令行入口，也可以使用 `aiscientist-mini`。

## 运行 Demo Study

Demo 不需要任何 API key，所有运行均为确定性的本地合成实验。典型流程：

```bash
# 创建或重置演示研究
python -m ai_scientist_mini demo memory-strategy --data-dir ./data/demo

# 启动本地 Dashboard（Tkinter 随 Python 提供）
python -m ai_scientist_mini serve --state ./data/demo/dashboard_state.json

# 导出研究报告、图表和审计记录
python -m ai_scientist_mini report --project memory-strategy --out ./artifacts/memory_strategy
```

也可以直接运行无窗口验收：

```bash
python -m ai_scientist_mini smoke-test --json
```

具体可用子命令以 `--help` 为准。Demo 预期产物：

```text
artifacts/
├── memory_strategy_study.json
├── memory_strategy_study.sqlite3
├── research_report.md
├── dashboard_snapshot.png
├── charts/
│   ├── mean_score_by_strategy.png
│   ├── success_rate_by_strategy.png
│   └── mean_score_by_round.svg
└── reproducibility_manifest.json
```

## 典型研究配置

```json
{
  "name": "Memory Strategy Study",
  "research_question": "哪一种 Memory 策略更适合长期 Agent？",
  "scope": "合成长期记忆任务；不代表真实世界结论",
  "constraints": ["仅本地实验", "不得调用付费 API"],
  "metrics": ["success_rate", "mean_score", "median_score", "std", "effect_size"],
  "budget": {
    "max_experiments": 10,
    "max_runs": 100,
    "max_runtime_seconds": 120,
    "estimated_api_cost": 0
  },
  "provider": "synthetic"
}
```

候选假设包括 `No Memory`、`Sliding Window`、`Summary`、`Episodic`、`Vector`。
每个 Experiment 和 Run 都保存配置、seed、数据哈希、环境、代码版本、结果和日志，
因此可以在关闭程序后恢复队列并继续研究。

演示同时输出人类可读 JSON 和事务性 SQLite checkpoint；两者只包含本地合成实验与
公开决策摘要，不包含模型私有 chain-of-thought。

## 项目结构

```text
src/ai_scientist_mini/
├── models.py        # Project、Hypothesis、Experiment、Run、Result 等模型
├── engine.py        # Scientist loop、设计、队列、选择、更新、分析
├── providers.py     # Synthetic / PythonFunction / MockLLM adapters
├── persistence.py   # SQLite/JSON checkpoint 与 schema 元数据
├── reporting.py     # 图表、报告、审计和 reproducibility manifest
├── ui/              # 本地 Dashboard 和 Hypothesis Graph
└── integrations/    # 版本化 JSON / CLI / loopback API 适配边界
tests/               # 单元、集成、回归和安全边界测试
```

模块之间只通过领域对象、Adapter、JSON、CLI 或本地 API 通信；当前项目不导入
Paper2Lab、LLM Lab、Agent Arena 或 Synthetic Benchmark Factory 的源码。

## 测试

```bash
python -m pytest -q
python -m pytest tests -m "not slow" -q
```

测试重点：scientific loop 闭环、预算/停止按钮、失败分类、去重、seed 可复现、重启恢复、
报告和审计追溯。测试不应要求网络或 API key；若某个测试需要外部 Provider，必须显式标记
`external`，并默认跳过。

## Windows 打包

在 Windows PowerShell 中执行：

```powershell
.\scripts\build_exe.ps1 -Clean -RunTests
```

产物写入 `dist\AIScientistMini\`，包括 `AIScientistMini.exe`、默认配置、README、
示例数据库和演示报告。打包脚本不会上传文件，也不会在构建时调用任何模型 API。

当前开发容器是 Linux，因此这里同时提供可直接验证的 Linux 打包脚本：

```bash
./scripts/build_linux.sh
./dist/AIScientistMini/AIScientistMini smoke-test --json
```

在 Windows 上运行同一份源码的 `scripts/build_exe.ps1` 会生成真正的
`AIScientistMini.exe`；仓库不把 Linux ELF 重命名为 Windows 文件，以免造成误导。

## 研究边界与免责声明

本软件是实验编排和证据追踪工具，不是自动化科学权威。Synthetic/Mock 结果只说明
规则和模拟数据在指定配置下的行为，不能替代真实数据、同行评审或领域专家判断。
报告必须列出限制、失败实验和下一步工作；用户在 Human Approval 模式下确认任何
未来可能产生外部资源消耗的操作。

更多说明见：

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：组件、数据流和扩展边界；
- [`SAFETY_REPRODUCIBILITY_AUDIT.md`](SAFETY_REPRODUCIBILITY_AUDIT.md)：安全、预算、复现、审计要求；
- [`TEST_STRATEGY.md`](TEST_STRATEGY.md)：验收矩阵和测试策略。
