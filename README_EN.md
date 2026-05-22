<div align="right">
# Sonarqube-remediation-ai-agent

  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Supervisor-FF6B35?style=flat-square" alt="LangGraph">
  <img src="https://img.shields.io/badge/LiteLLM-Unified%20LLM-7C3AED?style=flat-square" alt="LiteLLM">
  <img src="https://img.shields.io/badge/pgvector-RAG-16A34A?style=flat-square&logo=postgresql&logoColor=white" alt="pgvector">
  <img src="https://img.shields.io/badge/SonarQube-REST%20API-CB7A04?style=flat-square" alt="SonarQube">
  <img src="https://img.shields.io/badge/GitHub-PyGithub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub">

[简体中文](README.md) · English
</div>

<p align="center">
An AI-powered code remediation agent built on <strong>LangGraph Supervisor</strong>.<br>
Automatically reads open SonarQube issues, generates precise fix patches via LLM + RAG,<br>
validates fixes against SonarQube, and opens a GitHub Pull Request — fully automated.<br>
Supports crash recovery via resumable thread checkpoints.
</p>

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Development & Testing](#development--testing)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **End-to-end pipeline** | Issue fetch → RAG retrieval → LLM patch → validation → PR, fully automated |
| **RAG-augmented generation** | Retrieves 600+ SonarQube Java rule examples from pgvector to guide LLM fixes |
| **Multi-round retry loop** | Validator feedback drives up to `max_rounds` automatic retry cycles |
| **Resumable runs** | SQLite checkpoints persist full `AgentState`; resume any run via `--thread-id` |
| **Swappable LLMs** | LiteLLM unified interface — switch Claude / Azure GPT-4o with a single env var |
| **Dual embedding modes** | OpenAI `text-embedding-3-small` (online) or `all-MiniLM-L6-v2` (fully offline) |

---

## Architecture

> Full interactive diagram (draw.io): [docs/architecture-diagram-resume.html](docs/architecture-diagram-resume.html)

```mermaid
flowchart TD
    CLI(["🖥️ CLI Trigger\npython main.py run --project &lt;key&gt; --branch main --github-repo org/repo\n--thread-id &lt;uuid&gt;  ← optional, resume a prior run"])

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

### Dual-Database Design

| Database | Purpose | Location |
|----------|---------|----------|
| **SQLite** | LangGraph state checkpoints (enables resume) | `runs/agent_runs.db` |
| **PostgreSQL + pgvector** | SonarQube rule vector index (RAG retrieval) | Docker container / external service |

---

## Prerequisites

- **Python** 3.11+
- **Docker** — for running PostgreSQL + pgvector
- **SonarQube** — accessible instance with an API token (project read permission required)
- **GitHub Token** — with `repo` write scope
- **LLM API Key** — Anthropic API Key or Azure OpenAI credentials

---

## Installation

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd auto-sonarqube-reports-fix

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials (see Configuration section)
```

### 3. Start PostgreSQL + pgvector

```bash
# On first start, docker/init.sql runs automatically to enable the pgvector extension
docker compose up -d

# Confirm the container is healthy
docker compose ps
```

### 4. Initialize the RAG vector index

> **Required before first run.** Typically takes 2–5 minutes (600+ rules).

```bash
python -m rag.ingest --sonar-url http://your-sonar-host --token your_token
```

You only need to run this once. Re-run only if your SonarQube rule set changes significantly or you rebuild the Docker volume.

---

## Configuration

Copy `.env.example` to `.env` and populate the following:

```env
# ── SonarQube ──────────────────────────────────────────────────
SONAR_URL=http://sonar.internal          # SonarQube server URL
SONAR_TOKEN=your_sonarqube_token         # User token (project read permission)

# ── LLM — choose one ──────────────────────────────────────────
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-your-key

# Azure OpenAI alternative
# LLM_MODEL=azure/gpt-4o
# AZURE_API_KEY=your-azure-key
# AZURE_API_BASE=https://your-instance.openai.azure.com
# AZURE_API_VERSION=2024-02-01

# ── Embedding ──────────────────────────────────────────────────
OPENAI_API_KEY=sk-your-key               # Required when EMBEDDING_MODEL=openai
EMBEDDING_MODEL=openai                   # or: local (fully offline, no API key)

# ── PostgreSQL + pgvector ──────────────────────────────────────
PGVECTOR_DSN=postgresql://sonarrule:sonarrule@localhost:5432/sonarrule_rag

# ── GitHub ─────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_your-token             # repo scope required
GITHUB_REPO=myorg/payment-service       # format: org/repo

# ── Runtime parameters ─────────────────────────────────────────
MAX_ROUNDS=3                             # Max remediation rounds before forcing PR
REPO_LOCAL_PATH=/path/to/local/repo     # Absolute path to the already-cloned target repo
```

**Embedding model comparison:**

| Value | Model | Dimensions | Notes |
|-------|-------|-----------|-------|
| `openai` | text-embedding-3-small | 1536d | Requires `OPENAI_API_KEY`, better recall quality |
| `local` | all-MiniLM-L6-v2 | 384d | Fully offline, no API key needed, ideal for air-gapped environments |

---

## Usage

### Start a new fix run

```bash
python main.py run \
  --project com.example:payment-service \
  --branch main \
  --github-repo myorg/payment-service \
  --max-rounds 3
```

**Sample output:**

```
[agent] thread_id: a1b2c3d4-...  (use --thread-id a1b2c3d4-... to resume)
[agent] Starting run — thread_id=a1b2c3d4-...
[agent] Done. PR: https://github.com/myorg/payment-service/pull/42
```

### Resume an interrupted run

```bash
python main.py resume --thread-id a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### CLI argument reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--project` | SonarQube project key | **required** |
| `--branch` | Target branch | `main` |
| `--max-rounds` | Maximum remediation rounds | `3` |
| `--github-repo` | GitHub repository (`org/repo`) | **required** |
| `--thread-id` | Thread ID for resuming a run | auto-generated |

---

## How It Works

### RAG Retrieval

`rag/retriever.py` is invoked on every Remediator execution:

1. Concatenates the issue's `rule_id` + `rule_description` into a query string
2. Calls `EmbeddingModel.embed()` to produce a semantic vector
3. Runs cosine similarity search in pgvector, returning the Top-K most relevant rule documents
4. Injects retrieved documents into the LLM prompt as fix guidance

### Resumable Execution

Every `supervisor.invoke()` call triggers LangGraph's `SqliteSaver` to serialize the full `AgentState` to `runs/agent_runs.db`. When you resume with the same `thread_id`, LangGraph restores the last checkpoint and skips all already-completed nodes — no work is repeated.

### Multi-Round Retry

After the Validator triggers a SonarQube re-scan, the Supervisor inspects `remaining_issues`:
- If issues remain **and** `round < max_rounds` → route back to Remediator for a new patch attempt
- If all issues are resolved **or** `max_rounds` is exhausted → route to GitHubAgent to open the PR

---

## Project Structure

```
.
├── main.py                    # CLI entry point (run / resume subcommands)
├── state.py                   # AgentState TypedDict definition
├── config.py                  # Environment variable loading and validation
├── orchestrator/
│   └── supervisor.py          # LangGraph StateGraph + Supervisor routing
├── agents/
│   ├── issue_reader/          # SonarQube issue fetch subgraph
│   ├── remediation/           # RAG + LLM patch generation subgraph
│   ├── validation/            # Patch application + SonarQube validation subgraph
│   └── github/                # Branch push + PR creation subgraph
├── rag/
│   ├── embeddings.py          # EmbeddingModel (OpenAI / local dual-mode)
│   ├── retriever.py           # pgvector cosine similarity retrieval
│   └── ingest.py              # Offline rule vectorization init script
├── db/
│   └── sqlite.py              # LangGraph SQLite checkpoint wrapper
├── docker/
│   └── init.sql               # PostgreSQL init script (enables pgvector extension)
├── docker-compose.yml         # PostgreSQL + pgvector service definition
├── docs/
│   └── architecture-diagram-resume.html   # Interactive architecture diagram
├── tests/                     # Unit tests (fully mocked, no real services needed)
├── .env.example               # Environment variable template
└── requirements.txt           # Python dependency manifest
```

---

## Known Limitations

- **Java only** — The RAG index and IssueReader are designed for Java rules. Supporting other languages requires adjusting the rule filter logic in `rag/ingest.py`.
- **Pre-cloned repository required** — `REPO_LOCAL_PATH` must point to an already-cloned repo; the agent does not clone automatically.
- **pgvector requires initialization** — After rebuilding the Docker volume (`docker compose down -v`), you must re-run `python -m rag.ingest`.
- **LLM patch quality depends on prompt** — For complex multi-file changes (e.g. interface signature changes), the current single-file diff strategy may produce incomplete fixes.
- **SonarQube scan latency** — The Validator depends on a completed SonarQube scan. If CI has not triggered a scan, validation will read stale results.

---

## Development & Testing

### Run tests

```bash
# Run all unit tests
pytest tests/ -v

# Run a specific module
pytest tests/test_remediation.py -v
```

All tests are fully mocked — no real SonarQube, PostgreSQL, or GitHub connection is needed. Required environment variables are pre-configured in `tests/conftest.py`.

### Docker quick reference

```bash
# Start PostgreSQL in the background
docker compose up -d

# Stop services (data volume preserved)
docker compose down

# Destroy data volume (requires re-running rag/ingest)
docker compose down -v
```

---

> Chinese documentation: [README.md](README.md)
