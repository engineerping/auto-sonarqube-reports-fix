# Dual License Notice

This project is licensed under two distinct licenses:

1. **Personal / Non-Commercial Use**
   Free for personal, non-commercial purposes under the **MIT License**.

2. **Commercial Use**
   Requires a **paid commercial license**.
   See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for details.

<div align="center">

# Sonarqube-remediation-ai-agent

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Supervisor-FF6B35?logo=langgraph&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Unified%20LLM-7C3AED?logo=lightning&logoColor=white)](https://www.litellm.ai/)
[![pgvector](https://img.shields.io/badge/pgvector-RAG-16A34A?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![SonarQube](https://img.shields.io/badge/SonarQube-REST%20API-CB7A04?logo=sonarqube&logoColor=white)](https://www.sonarsource.com/products/sonarqube/)
[![GitHub](https://img.shields.io/badge/GitHub-PyGithub-181717?logo=github&logoColor=white)](https://github.com/)

[English](README_en.md) · 简体中文
</div>

---

1. 基于 **LangGraph Supervisor** 模式的 AI 自动代码修复代理。
1. 自动读取 SonarQube 检测到的 Java 代码问题，通过 LLM + RAG 生成精准修复补丁，
1. 验证修复结果后，自动在 GitHub 上创建 Pull Request。
1. 支持断点续跑。

---

## 目录

- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [环境要求](#环境要求)
- [安装与启动](#安装与启动)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [工作原理](#工作原理)
- [项目结构](#项目结构)
- [已知限制](#已知限制)
- [开发与测试](#开发与测试)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **全自动修复流水线** | SonarQube 问题读取 → RAG 检索 → LLM 生成补丁 → 验证 → 自动提 PR |
| **RAG 增强生成** | 从 pgvector 召回 600+ 条 SonarQube Java 规则，精准辅助 LLM 理解修复方向 |
| **多轮重试循环** | Validator 反馈驱动，最多 `max_rounds` 轮自动重试，直到问题解决 |
| **断点续跑** | SQLite 检查点持久化 AgentState，中断后凭 `thread_id` 无缝恢复 |
| **模型可切换** | 通过 LiteLLM 统一接口，一行配置切换 Claude / Azure GPT-4o 等模型 |
| **Embedding 双模式** | 支持 OpenAI `text-embedding-3-small`（在线）或 `all-MiniLM-L6-v2`（完全离线）|

---

## 架构设计

> 交互式完整架构图（draw.io）：[docs/architecture-diagram-resume.html](docs/architecture-diagram-resume.html)

```mermaid
flowchart TD
    CLI(["🖥️ CLI Trigger\npython main.py run --project &lt;key&gt; --branch main --github-repo org/repo\n--thread-id &lt;uuid&gt;  ← 可选，用于断点续跑"])

    subgraph SG["⚙️  LangGraph Supervisor Graph  ·  StateGraph · SqliteSaver · resumable · max_rounds retry loop"]
        direction LR
        A1["📋 ① IssueReader\nSonarQube REST · Subgraph\n· Fetch open issues\n· Fetch rule metadata\n· Fetch source context\n· Emit Issue[] → state"]
        A2["🔧 ② Remediator\npgvector · LiteLLM · Subgraph\n· RAG: retrieve rules\n· LiteLLM: generate fix\n· Indent-preserving patch\n· Emit Fix[] → state"]
        A3["☑️ ③ Validator\nsonar-scanner CLI · Subgraph\n· Apply unified diffs\n· Trigger sonar-scanner\n· Poll CE task status\n· Emit ValidationResult"]
        A4["🐙 ④ GitHubAgent\nPyGithub · Subgraph\n· Create feature branch\n· Commit patched files\n· Open Pull Request\n· Emit pr_url → state"]
        A1 --> A2 --> A3 --> A4
        A3 -- "remaining > 0\n& round < N  ↩ Retry" --> A2
    end

    RAG[("🗄️ pgvector\nPostgreSQL + pgvector\n600+ Java rules\nOpenAI embed-3-small\ncosine top-k")]
    LLM[("🤖 LiteLLM\nUnified LLM interface\nClaude / Azure GPT-4o\nvia LLM_MODEL env")]
    SQLITE[("💾 SQLite\nLangGraph SqliteSaver\nAgentState per step\nruns/agent_runs.db")]
    SONAR[["🔎 SonarQube Server\non-premise / private cloud\n/api/issues · /api/rules\n/api/sources · /api/ce/task"]]
    GH[["🐙 GitHub\ntarget repository\nPyGithub API"]]

    CLI --> SG
    A2 -.-> RAG & LLM
    SG -. "checkpoint / step" .-> SQLITE
    A1 & A3 -. "REST API" .-> SONAR
    A4 -. "push branch + PR" .-> GH

    classDef agent fill:#e0f2f1,stroke:#4db6ac,color:#1a1a1a
    classDef ragNode fill:#e8f5e9,stroke:#66bb6a,color:#1a1a1a
    classDef llmNode fill:#ede7f6,stroke:#9575cd,color:#1a1a1a
    classDef dbNode fill:#fff3e0,stroke:#ffb74d,color:#1a1a1a
    classDef extNode fill:#fff8f1,stroke:#ffa726,color:#1a1a1a
    classDef cliNode fill:#e8e8e8,stroke:#bdbdbd,color:#1a1a1a

    class A1,A2,A3,A4 agent
    class RAG ragNode
    class LLM llmNode
    class SQLITE dbNode
    class SONAR,GH extNode
    class CLI cliNode
```

### 双数据库设计

| 数据库 | 用途 | 位置 |
|--------|------|------|
| **SQLite** | LangGraph 状态检查点（支持断点续跑） | `runs/agent_runs.db` |
| **PostgreSQL + pgvector** | SonarQube 规则向量索引（RAG 检索） | Docker 容器 / 外部服务 |

---

## 环境要求

- **Python** 3.11+
- **Docker** — 用于启动 PostgreSQL + pgvector
- **SonarQube** — 可访问的实例，含 API Token（需项目读权限）
- **GitHub Token** — 需要 `repo` 写权限
- **LLM API Key** — Anthropic API Key 或 Azure OpenAI 配置

---

## 安装与启动

### 1. 克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd auto-sonarqube-reports-fix

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入各项凭据（详见「配置说明」）
```

### 3. 启动 PostgreSQL + pgvector

```bash
# 首次启动会自动执行 docker/init.sql，创建 pgvector 扩展
docker compose up -d

# 确认容器健康
docker compose ps
```

### 4. 初始化 RAG 向量索引

> 首次运行前**必须**执行。通常需要 2–5 分钟（600+ 条规则）。

```bash
python -m rag.ingest --sonar-url http://your-sonar-host --token your_token
```

完成后无需重复执行，除非 SonarQube 规则库有重大更新或重建了 Docker 数据卷。

---

## 配置说明

复制 `.env.example` 为 `.env` 并填写：

```env
# ── SonarQube ──────────────────────────────────────────
SONAR_URL=http://sonar.internal          # SonarQube 服务地址
SONAR_TOKEN=your_sonarqube_token         # 用户 Token（需有项目读权限）

# ── LLM（二选一）────────────────────────────────────────
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-your-key

# Azure OpenAI 替代方案
# LLM_MODEL=azure/gpt-4o
# AZURE_API_KEY=your-azure-key
# AZURE_API_BASE=https://your-instance.openai.azure.com
# AZURE_API_VERSION=2024-02-01

# ── Embedding ──────────────────────────────────────────
OPENAI_API_KEY=sk-your-key               # 使用 OpenAI embedding 时必填
EMBEDDING_MODEL=openai                   # 或 local（离线，无需 API Key）

# ── PostgreSQL + pgvector ──────────────────────────────
PGVECTOR_DSN=postgresql://sonarrule:sonarrule@localhost:5432/sonarrule_rag

# ── GitHub ─────────────────────────────────────────────
GITHUB_TOKEN=ghp_your-token             # 需要 repo 权限
GITHUB_REPO=myorg/payment-service       # 格式：org/repo

# ── 运行参数 ───────────────────────────────────────────
MAX_ROUNDS=3                             # 最大修复轮次（超过后强制提 PR）
REPO_LOCAL_PATH=/path/to/local/repo     # 本地已克隆的目标仓库绝对路径
```

**Embedding 模型对比：**

| 值 | 模型 | 向量维度 | 说明 |
|----|------|----------|------|
| `openai` | text-embedding-3-small | 1536d | 需要 `OPENAI_API_KEY`，召回质量更高 |
| `local` | all-MiniLM-L6-v2 | 384d | 完全离线，无需 API Key，适合内网环境 |

---

## 使用方法

### 启动新的修复任务

```bash
python main.py run \
  --project com.example:payment-service \
  --branch main \
  --github-repo myorg/payment-service \
  --max-rounds 3
```

**输出示例：**

```
[agent] thread_id: a1b2c3d4-...  (use --thread-id a1b2c3d4-... to resume)
[agent] Starting run — thread_id=a1b2c3d4-...
[agent] Done. PR: https://github.com/myorg/payment-service/pull/42
```

### 恢复中断的任务

```bash
python main.py resume --thread-id a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--project` | SonarQube 项目 Key | **必填** |
| `--branch` | 目标分支 | `main` |
| `--max-rounds` | 最大修复轮次 | `3` |
| `--github-repo` | GitHub 仓库（`org/repo`） | **必填** |
| `--thread-id` | 指定 thread ID（用于断点续跑） | 自动生成 |

---

## 工作原理

### RAG 检索流程

`rag/retriever.py` 在每次 Remediator 执行时被调用：

1. 将 SonarQube 问题的 `rule_id` + `rule_description` 拼接成查询文本
2. 调用 `EmbeddingModel.embed()` 获取语义向量
3. 在 pgvector 中执行余弦相似度检索，返回 Top-K 最相关规则文档
4. 将检索结果注入 LLM Prompt，提供精准的修复参考

### 断点续跑原理

每次 `supervisor.invoke()` 调用时，LangGraph 通过 `SqliteSaver` 将完整的 `AgentState` 序列化到 `runs/agent_runs.db`。恢复时传入相同 `thread_id`，LangGraph 自动从上次检查点恢复，跳过已完成节点。

### 多轮重试机制

Validator 完成 SonarQube 重扫后，Supervisor 检查 `remaining_issues` 数量：
- 若仍有未解决问题 **且** `round < max_rounds`，回到 Remediator 重新生成补丁
- 超出轮次或全部解决，路由到 GitHubAgent 提交 PR

---

## 项目结构

```
.
├── main.py                    # CLI 入口（run / resume 子命令）
├── state.py                   # AgentState TypedDict 定义
├── config.py                  # 环境变量读取与校验
├── orchestrator/
│   └── supervisor.py          # LangGraph StateGraph + Supervisor 路由
├── agents/
│   ├── issue_reader/          # SonarQube 问题读取 Subgraph
│   ├── remediation/           # RAG + LLM 修复生成 Subgraph
│   ├── validation/            # 补丁应用 + SonarQube 验证 Subgraph
│   └── github/                # GitHub 分支推送 + PR 创建 Subgraph
├── rag/
│   ├── embeddings.py          # EmbeddingModel（OpenAI / 本地双模式）
│   ├── retriever.py           # pgvector 余弦相似度检索
│   └── ingest.py              # 规则库向量化离线初始化脚本
├── db/
│   └── sqlite.py              # LangGraph SQLite 检查点封装
├── docker/
│   └── init.sql               # PostgreSQL 初始化（启用 pgvector 扩展）
├── docker-compose.yml         # PostgreSQL + pgvector 服务定义
├── docs/
│   └── architecture-diagram-resume.html   # 交互式架构图（浏览器打开）
├── tests/                     # 单元测试（全 Mock，无需真实服务）
├── .env.example               # 环境变量配置模板
└── requirements.txt           # Python 依赖清单
```

---

## 已知限制

- **仅支持 Java 项目** — RAG 向量索引和 IssueReader 均针对 Java 规则设计；其他语言需调整 `rag/ingest.py` 的规则过滤逻辑
- **本地仓库需预先克隆** — `REPO_LOCAL_PATH` 必须指向已克隆好的仓库，Agent 不会自动 clone
- **pgvector 需首次初始化** — 重建 Docker 数据卷（`docker compose down -v`）后，需重新执行 `python -m rag.ingest`
- **LLM 生成质量依赖 Prompt** — 对于复杂的多文件修复（如接口变更），当前单文件 diff 策略可能产生不完整的修复
- **SonarQube 扫描延迟** — Validator 依赖 SonarQube 完成扫描后的结果；若 CI 扫描未触发，验证将得到旧数据

---

## 开发与测试

### 运行测试

```bash
# 运行全部单元测试
pytest tests/ -v

# 运行特定模块的测试
pytest tests/test_remediation.py -v
```

所有测试均使用 Mock，无需真实的 SonarQube、PostgreSQL 或 GitHub 连接。`tests/conftest.py` 中预设了所需的环境变量。

### Docker 常用命令

```bash
# 启动 PostgreSQL（后台运行）
docker compose up -d

# 停止服务（保留数据卷）
docker compose down

# 彻底清除数据（需重新执行 rag/ingest）
docker compose down -v
```

---

> 英文文档请参阅 [README_EN.md](README_EN.md)
