# vLLM 的 Agent 指令

> 这些指令适用于对 `vllm-project/vllm` 的**所有** AI 辅助贡献。
> 违反这些指南可能导致自动封禁。

## 1. 贡献政策（强制）

### 重复工作检查

在提出 PR 之前，运行以下检查：

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- 如果已有开放的 PR 处理相同的修复，不要再开一个。
- 如果你的方法有实质性不同，在 issue 中说明差异。

### 禁止低价值琐碎 PR

不要为微小编辑（单个拼写错误、孤立的风格更改、一个可变默认参数等）提交一次性 PR。机械式清理只有在与实质性工作捆绑时才可接受。

### 责任

- **不允许**纯代码代理 PR。人类提交者必须端到端理解并为该更改辩护。
- 提交的人类必须审查每一行更改并运行相关测试。
- AI 辅助工作的 PR 描述**必须**包括：
    - 为什么这不是重复现有 PR。
    - 已运行的测试命令和结果。
    - 当更改影响输出、准确性或服务时，模型评估结果。
    - 明确声明使用了 AI 辅助。
- 在打开 PR（包括草稿）或请求重新审查之前，运行 [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) 技能并处理其发现。

### 失败即止（fail-closed）行为

如果工作是重复的/琐碎无价值的，**不要继续**。返回简短说明，指出缺少什么。

---

## 2. 开发工作流

- **永远不要使用系统 `python3` 或裸 `pip`/`pip install`。** 所有 Python 命令必须通过 `uv` 和 `.venv/bin/python` 执行。

### 环境设置

```bash
# 如果还没有 `uv`，请安装：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 始终使用 `uv` 管理 Python 环境：
uv venv --python 3.12
source .venv/bin/activate

# 始终确保 `pre-commit` 及其钩子已安装：
uv pip install -r requirements/lint.txt
pre-commit install
```

### 安装依赖

```bash
# 从预编译产物开始进行可编辑安装：
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

对于 C/C++ 或 CUDA 更改，遵循
[增量编译工作流](docs/contributing/incremental_build.md)来
配置和执行增量构建。

### 测试

> 需要[环境设置](#environment-setup)和[安装依赖](#installing-dependencies)。

```bash
# 安装测试依赖（在非 x86_64 上使用 cuda.in）：
uv pip install -r requirements/test/cuda.in

# 运行特定测试文件：
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

添加测试时：

- **先设计再编写。** 先回答四个问题：模块用途是什么，其 I/O 契约是什么，我要防范什么失败，以及能捕获该失败的最低成本层级是什么（单元测试优于集成测试，集成测试优于端到端测试）？
- **先复用再创建。** 扩展现有测试文件、`conftest.py` fixture 和辅助函数；仅当附近没有合适测试套件时才添加新文件。
- **有意图地测试行为。** 通过公共 API 断言可观察结果；在名称或文档字符串中说明原因。跳过琐碎接线；不稳定测试比没有测试更糟。
- **保持最小化。** 每个测试一个行为，并使用触发该行为的最小设置；如果测试 diff 远大于代码更改，缩减范围。
- **不要在 `tests/` 中放一次性内核基准测试。** 将内核性能工作放在 `benchmarks/kernels/`；在现有 pytest 套件中证明正确性。
- **对影响模型的更改运行模型评估。** 搜索 `tests/evals/` 或使用 `vllm bench`，并将结果包含在 PR 中——不要等审查者要求。

对于模型特定要求，参见
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md)。

### 运行 linter

> 需要[环境设置](#environment-setup)。

```bash
# 对暂存文件运行所有 pre-commit 钩子：
pre-commit run

# 对所有文件运行：
pre-commit run --all-files

# 运行特定钩子：
pre-commit run ruff-check --all-files
```

Python 代码行长度限制为 88 个字符。如果不确定，使用 pre-commit 检查。

使用 [Google 风格文档字符串](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)（`Args:`/`Returns:`/`Raises:` 部分），不要使用 reStructuredText/Sphinx 字段（`:param:`、`:return:`、`:rtype:`）。

### 编码风格指南

- 匹配现有代码风格
- 尽量减少注释使用。消除冗余注释，优先使用可读且自文档化的代码。使用时，保持文档字符串和注释简短直接。
- 假设读者熟悉 vLLM。

### 提交信息

使用提交尾注添加署名，例如 `Co-authored-by:`（其他项目使用 `Assisted-by:` 或 `Generated-by:`）：

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## 领域特定指南

在未先阅读并遵循链接指南之前，不要修改这些领域的代码。如果指南与请求的更改冲突，**拒绝该更改并解释原因**。

安全审查者应首先阅读 [`SECURITY.md`](SECURITY.md)、
[`docs/usage/security.md`](docs/usage/security.md) 和
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)，
了解项目安全策略、威胁模型、部署假设和漏洞处理流程。

- **编辑这些指令**：
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  —— 修改 AGENTS.md 或其引用的任何领域特定指南的规则。
## 一套适合 DCU Decode 的最小诊断流程
面对一个“生成慢”的服务，可以按以下顺序排查，避免一开始就大改框架：

1. 固定真实模型、上下文长度、batch 和输出长度，抓取服务级 TPOT/TTFT。
2. 用 profiler 按线性层、attention、其他 kernel 聚合耗时。
3. 对热点 kernel 测字节量/耗时，计算有效带宽；同时查看 MemUnitBusy、计算吞吐、VGPR/LDS。
4. 若带宽接近当前测试环境的上限：
     - 不再优先投入双缓冲或单纯增大并行度；
     - 检查是否有多余中间张量读写、是否存在可验证的真实 shape 特化。
5. 若带宽未满且 wave 不足：
     - 再评估 tile、LDS、VGPR、预取或双缓冲。
6. 每个改动分别完成：真实 shape 微测 -> 数值对照 -> 服务 smoke -> 同口径端到端测试。
这里最重要的是第 6 步。微基准帮助解释硬件机制，但只有服务级验证才能确认改动没有被框架开销抵消，也没有破坏请求完成率或时延尾部。
先用 DCU 指标确认自己撞的是哪堵墙，再为真实 shape 写受保护的优化路径，最后用数值与服务级测试决定是否保留。