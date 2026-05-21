# SonarQube AI Auto-Fix Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph multi-agent system that reads SonarQube issues, auto-generates indentation-preserving Java code fixes via RAG + LiteLLM, validates with a re-scan, and raises a GitHub PR.

**Architecture:** A LangGraph Supervisor graph routes work across 4 independently compiled subgraphs (IssueReaderAgent, RemediationAgent, ValidationAgent, GitHubAgent). A central `router` node reads `AgentState` to decide which subgraph runs next, enabling multi-round fix loops up to `max_rounds`.

**Tech Stack:** Python 3.11+, LangGraph 0.2+, LiteLLM, pgvector (PostgreSQL), SQLite (LangGraph checkpointing), SonarQube REST API, PyGithub, sentence-transformers / OpenAI embeddings.

---

## File Map

| Path | Responsibility |
|------|---------------|
| `requirements.txt` | All Python dependencies |
| `docker-compose.yml` | PostgreSQL + pgvector container |
| `.env.example` | Environment variable template |
| `config.py` | Load env vars into typed constants |
| `state.py` | `Issue`, `Fix`, `ValidationResult`, `AgentState` TypedDicts |
| `db/sqlite.py` | LangGraph `SqliteSaver` initialisation |
| `rag/embeddings.py` | `EmbeddingModel` — wraps OpenAI or local sentence-transformers |
| `rag/retriever.py` | `RAGRetriever` — cosine search against `sonar_rules` in pgvector |
| `rag/ingest.py` | CLI: fetch Java rules from SonarQube API, embed, upsert to pgvector |
| `mcp/sonarqube_mcp.py` | `SonarQubeClient` — REST wrapper for all SonarQube API calls |
| `agents/issue_reader/tools.py` | `fetch_issues`, `fetch_rule`, `fetch_source` LangChain tools |
| `agents/issue_reader/agent.py` | `IssueReaderAgent` subgraph + `issue_reader_node` |
| `agents/remediation/patch.py` | `detect_base_indent`, `apply_patch`, `create_unified_diff` |
| `agents/remediation/tools.py` | `read_file`, `rag_retrieve`, `call_llm`, `write_fix` tools |
| `agents/remediation/agent.py` | `RemediationAgent` subgraph + `remediation_node` |
| `agents/validation/tools.py` | `apply_diffs`, `trigger_scan`, `poll_task`, `check_issues` tools |
| `agents/validation/agent.py` | `ValidationAgent` subgraph + `validation_node` |
| `agents/github/tools.py` | `create_branch`, `commit_files`, `create_pr` tools |
| `agents/github/agent.py` | `GitHubAgent` subgraph + `github_node` |
| `orchestrator/supervisor.py` | `route()` function + `build_supervisor()` |
| `main.py` | `argparse` CLI entry point |
| `tests/conftest.py` | `make_state` pytest fixture |
| `tests/test_patch.py` | Unit tests for indentation logic |
| `tests/test_supervisor.py` | Unit tests for routing logic |
| `tests/test_issue_reader.py` | IssueReaderAgent with mocked SonarQube client |
| `tests/test_remediation.py` | RemediationAgent with mocked LLM + RAG |
| `tests/test_validation.py` | ValidationAgent with mocked scanner + API |

---

## Task 1: Project Bootstrap

**Files:**
- Create: `requirements.txt`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: all `__init__.py` files

- [ ] **Step 1: Create requirements.txt**

```
langgraph>=0.2.0
langchain-core>=0.2.0
litellm>=1.40.0
anthropic>=0.30.0
openai>=1.30.0
pgvector>=0.2.5
psycopg2-binary>=2.9.9
PyGithub>=2.3.0
sentence-transformers>=3.0.0
requests>=2.31.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: sonarrule
      POSTGRES_PASSWORD: sonarrule
      POSTGRES_DB: sonarrule_rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 3: Create .env.example**

```
SONAR_URL=http://sonar.internal
SONAR_TOKEN=your_sonarqube_token

LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-your-key

# Azure OpenAI alternative:
# LLM_MODEL=azure/gpt-4o
# AZURE_API_KEY=your-azure-key
# AZURE_API_BASE=https://your-instance.openai.azure.com
# AZURE_API_VERSION=2024-02-01

OPENAI_API_KEY=sk-your-key
EMBEDDING_MODEL=openai

PGVECTOR_DSN=postgresql://sonarrule:sonarrule@localhost:5432/sonarrule_rag

GITHUB_TOKEN=ghp_your-token
GITHUB_REPO=myorg/payment-service

MAX_ROUNDS=3
REPO_LOCAL_PATH=/path/to/local/cloned/repo
```

- [ ] **Step 4: Create package __init__.py files**

Create empty `__init__.py` in: `agents/`, `agents/issue_reader/`, `agents/remediation/`, `agents/validation/`, `agents/github/`, `orchestrator/`, `rag/`, `db/`, `mcp/`, `tests/`.

```bash
mkdir -p agents/issue_reader agents/remediation agents/validation agents/github
mkdir -p orchestrator rag db mcp tests runs
touch agents/__init__.py agents/issue_reader/__init__.py \
      agents/remediation/__init__.py agents/validation/__init__.py \
      agents/github/__init__.py orchestrator/__init__.py \
      rag/__init__.py db/__init__.py mcp/__init__.py tests/__init__.py
```

- [ ] **Step 5: Install dependencies and start postgres**

```bash
pip install -r requirements.txt
docker-compose up -d
```

Expected: `docker ps` shows `pgvector/pgvector:pg16` running on port 5432.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt docker-compose.yml .env.example \
        agents orchestrator rag db mcp tests
git commit -m "chore: bootstrap project structure and dependencies"
```

---

## Task 2: Shared State Types

**Files:**
- Create: `state.py`

- [ ] **Step 1: Write state.py**

```python
# state.py
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Issue(TypedDict):
    issue_key: str
    rule_id: str
    rule_description: str
    remediation_guidance: str
    severity: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str


class Fix(TypedDict):
    issue_key: str
    file_path: str
    unified_diff: str
    original_snippet: str
    fixed_snippet: str


class ValidationResult(TypedDict):
    resolved_issues: List[str]
    remaining_issues: List[Issue]
    all_resolved: bool
    round_number: int


class AgentState(TypedDict):
    project_key: str
    branch: str
    issues: List[Issue]
    fixes: List[Fix]
    validation_result: Optional[ValidationResult]
    round_number: int
    max_rounds: int
    pr_url: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from state import AgentState, Issue, Fix, ValidationResult; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add state.py
git commit -m "feat: add shared AgentState TypedDicts"
```

---

## Task 3: SQLite Checkpointer

**Files:**
- Create: `db/sqlite.py`

- [ ] **Step 1: Write db/sqlite.py**

```python
# db/sqlite.py
import os
from langgraph.checkpoint.sqlite import SqliteSaver

os.makedirs("runs", exist_ok=True)

checkpointer = SqliteSaver.from_conn_string("runs/agent_runs.db")
```

- [ ] **Step 2: Verify checkpointer instantiates**

```bash
python -c "from db.sqlite import checkpointer; print(type(checkpointer))"
```

Expected: `<class 'langgraph.checkpoint.sqlite.SqliteSaver'>`

- [ ] **Step 3: Commit**

```bash
git add db/sqlite.py
git commit -m "feat: add SQLite checkpointer for LangGraph state persistence"
```

---

## Task 4: Embedding Model Wrapper

**Files:**
- Create: `rag/embeddings.py`

- [ ] **Step 1: Write rag/embeddings.py**

```python
# rag/embeddings.py
import os
from typing import List


class EmbeddingModel:
    """Wraps OpenAI text-embedding-3-small or local sentence-transformers.

    Set EMBEDDING_MODEL=openai (default) or EMBEDDING_MODEL=local.
    """

    def __init__(self):
        self.mode = os.getenv("EMBEDDING_MODEL", "openai")
        self._local_model = None
        self.dimension = 1536

        if self.mode == "local":
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = 384

    def embed(self, text: str) -> List[float]:
        if self.mode == "local":
            return self._local_model.encode(text).tolist()
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "local":
            return self._local_model.encode(texts).tolist()
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
```

- [ ] **Step 2: Verify import**

```bash
python -c "from rag.embeddings import EmbeddingModel; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/embeddings.py
git commit -m "feat: add EmbeddingModel wrapper (OpenAI + local fallback)"
```

---

## Task 5: pgvector RAG Retriever

**Files:**
- Create: `rag/retriever.py`

- [ ] **Step 1: Write rag/retriever.py**

```python
# rag/retriever.py
import os
from typing import List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from rag.embeddings import EmbeddingModel


CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sonar_rules (
    id          SERIAL PRIMARY KEY,
    rule_key    VARCHAR(100) UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    remediation TEXT,
    severity    VARCHAR(20),
    embedding   vector(1536)
);

CREATE INDEX IF NOT EXISTS sonar_rules_embedding_idx
    ON sonar_rules USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class RAGRetriever:
    def __init__(self):
        self.dsn = os.environ["PGVECTOR_DSN"]
        self.embedder = EmbeddingModel()
        self._ensure_table()

    def _conn(self):
        conn = psycopg2.connect(self.dsn)
        register_vector(conn)
        return conn

    def _ensure_table(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()

    def upsert(self, rule_key: str, name: str, description: str,
               remediation: str, severity: str, embedding: List[float]):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sonar_rules
                        (rule_key, name, description, remediation, severity, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (rule_key) DO UPDATE SET
                        name        = EXCLUDED.name,
                        description = EXCLUDED.description,
                        remediation = EXCLUDED.remediation,
                        severity    = EXCLUDED.severity,
                        embedding   = EXCLUDED.embedding
                """, (rule_key, name, description, remediation, severity, embedding))
            conn.commit()

    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        embedding = self.embedder.embed(query_text)
        with self._conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT rule_key, name, description, remediation, severity
                    FROM sonar_rules
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding, top_k))
                return [dict(row) for row in cur.fetchall()]
```

- [ ] **Step 2: Verify import**

```bash
python -c "from rag.retriever import RAGRetriever; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag/retriever.py
git commit -m "feat: add RAGRetriever with pgvector cosine search"
```

---

## Task 6: RAG Ingest Script

**Files:**
- Create: `rag/ingest.py`

- [ ] **Step 1: Write rag/ingest.py**

```python
# rag/ingest.py
"""Offline script: fetch all Java rules from SonarQube and index into pgvector."""
import argparse
import os
import sys
import requests
from dotenv import load_dotenv
from rag.retriever import RAGRetriever
from rag.embeddings import EmbeddingModel

load_dotenv()


def fetch_java_rules(sonar_url: str, token: str):
    rules = []
    page = 1
    while True:
        resp = requests.get(
            f"{sonar_url}/api/rules/search",
            params={"languages": "java", "ps": 500, "p": page},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rules.extend(data["rules"])
        print(f"  Fetched page {page}: {len(data['rules'])} rules "
              f"(total so far: {len(rules)})")
        if len(rules) >= data["total"]:
            break
        page += 1
    return rules


def build_document(rule: dict) -> str:
    name = rule.get("name", "")
    desc = rule.get("htmlDesc", rule.get("mdDesc", ""))
    # Strip basic HTML tags for cleaner embedding text
    import re
    desc = re.sub(r"<[^>]+>", " ", desc).strip()
    effort = rule.get("remFnBaseEffort", "")
    return f"{rule['key']}: {name}. {desc}. Fix effort: {effort}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonar-url", default=os.getenv("SONAR_URL"))
    parser.add_argument("--token", default=os.getenv("SONAR_TOKEN"))
    args = parser.parse_args()

    if not args.sonar_url or not args.token:
        print("ERROR: --sonar-url and --token are required (or set SONAR_URL/SONAR_TOKEN)")
        sys.exit(1)

    print("Fetching Java rules from SonarQube...")
    rules = fetch_java_rules(args.sonar_url, args.token)
    print(f"Total rules fetched: {len(rules)}")

    retriever = RAGRetriever()
    embedder = EmbeddingModel()

    print("Embedding and upserting rules into pgvector (batch of 50)...")
    batch_size = 50
    for i in range(0, len(rules), batch_size):
        batch = rules[i:i + batch_size]
        docs = [build_document(r) for r in batch]
        embeddings = embedder.embed_batch(docs)
        for rule, embedding in zip(batch, embeddings):
            retriever.upsert(
                rule_key=rule["key"],
                name=rule.get("name", ""),
                description=rule.get("htmlDesc", rule.get("mdDesc", "")),
                remediation=rule.get("remFnBaseEffort", ""),
                severity=rule.get("severity", ""),
                embedding=embedding,
            )
        print(f"  Upserted rules {i + 1}–{min(i + batch_size, len(rules))}")

    print(f"Done. {len(rules)} rules indexed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add rag/ingest.py
git commit -m "feat: add RAG ingest script for SonarQube Java rules"
```

---

## Task 7: SonarQube REST Client

**Files:**
- Create: `mcp/sonarqube_mcp.py`

*Note: This implements the SonarQube MCP Server interface as a direct REST client, matching the tool contract of the `sonarqube-agent-plugins` MCP server.*

- [ ] **Step 1: Write mcp/sonarqube_mcp.py**

```python
# mcp/sonarqube_mcp.py
import os
import time
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class SonarQubeClient:
    """REST client matching the SonarQube MCP Server tool interface."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = (url or os.environ["SONAR_URL"]).rstrip("/")
        self.token = token or os.environ["SONAR_TOKEN"]
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _get(self, path: str, params: dict = None) -> dict:
        resp = self.session.get(f"{self.url}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_issues(self, project_key: str, branch: str,
                   severities: Optional[List[str]] = None) -> List[Dict]:
        params = {
            "projectKeys": project_key,
            "branch": branch,
            "statuses": "OPEN",
            "ps": 500,
        }
        if severities:
            params["severities"] = ",".join(severities)
        data = self._get("/api/issues/search", params)
        return data.get("issues", [])

    def get_rule(self, rule_key: str) -> Dict:
        data = self._get("/api/rules/show", {"key": rule_key})
        return data.get("rule", {})

    def get_source(self, component_key: str,
                   from_line: int, to_line: int) -> str:
        data = self._get("/api/sources/lines", {
            "key": component_key,
            "from": max(1, from_line),
            "to": to_line,
        })
        lines = data.get("sources", [])
        return "\n".join(src.get("code", "") for src in lines)

    def trigger_scan(self, project_key: str, sonar_url: str,
                     repo_path: str) -> str:
        """Run sonar-scanner CLI and return the CE task ID."""
        import subprocess
        result = subprocess.run(
            [
                "sonar-scanner",
                f"-Dsonar.projectKey={project_key}",
                f"-Dsonar.host.url={sonar_url}",
                f"-Dsonar.token={self.token}",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sonar-scanner failed:\n{result.stderr}")
        for line in result.stdout.splitlines():
            if "task?id=" in line:
                return line.split("task?id=")[-1].strip()
        raise RuntimeError("Could not extract CE task ID from sonar-scanner output")

    def poll_task(self, task_id: str, timeout: int = 300) -> Dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self._get("/api/ce/task", {"id": task_id})
            task = data.get("task", {})
            status = task.get("status")
            if status == "SUCCESS":
                return task
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"SonarQube task {task_id} ended with status {status}")
            time.sleep(10)
        raise TimeoutError(f"SonarQube task {task_id} did not complete in {timeout}s")

    def check_issues_resolved(self, project_key: str, branch: str,
                               issue_keys: List[str]) -> Dict:
        """Return which of the given issue keys are still OPEN."""
        all_open = self.get_issues(project_key, branch)
        open_keys = {i["key"] for i in all_open}
        resolved = [k for k in issue_keys if k not in open_keys]
        remaining_raw = [i for i in all_open if i["key"] in issue_keys]
        return {"resolved": resolved, "remaining_raw": remaining_raw}
```

- [ ] **Step 2: Verify import**

```bash
python -c "from mcp.sonarqube_mcp import SonarQubeClient; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add mcp/sonarqube_mcp.py
git commit -m "feat: add SonarQube REST client (MCP Server interface)"
```

---

## Task 8: IssueReaderAgent Subgraph

**Files:**
- Create: `agents/issue_reader/tools.py`
- Create: `agents/issue_reader/agent.py`
- Create: `tests/test_issue_reader.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_issue_reader.py
from unittest.mock import patch, MagicMock
from agents.issue_reader.agent import issue_reader_node
from tests.conftest import make_state


@patch("mcp.sonarqube_mcp.SonarQubeClient.get_issues")
@patch("mcp.sonarqube_mcp.SonarQubeClient.get_rule")
@patch("mcp.sonarqube_mcp.SonarQubeClient.get_source")
def test_issue_reader_node_maps_api_response(mock_source, mock_rule, mock_issues):
    mock_issues.return_value = [
        {
            "key": "AX123",
            "rule": "java:S1192",
            "component": "test:proj:src/main/java/PaymentService.java",
            "textRange": {"startLine": 10, "endLine": 10},
            "severity": "MAJOR",
            "message": "String literal duplicated",
        }
    ]
    mock_rule.return_value = {
        "key": "java:S1192",
        "name": "String literals should not be duplicated",
        "htmlDesc": "Duplicating a string literal ...",
        "remFnBaseEffort": "5min",
    }
    mock_source.return_value = '        String status = "PENDING";'

    state = make_state()
    result = issue_reader_node(state)

    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["issue_key"] == "AX123"
    assert issue["rule_id"] == "java:S1192"
    assert issue["severity"] == "MAJOR"
    assert issue["line_start"] == 10
    assert "PENDING" in issue["code_snippet"]


@patch("mcp.sonarqube_mcp.SonarQubeClient.get_issues")
def test_issue_reader_returns_empty_when_no_issues(mock_issues):
    mock_issues.return_value = []
    state = make_state()
    result = issue_reader_node(state)
    assert result["issues"] == []
```

- [ ] **Step 2: Create conftest.py**

```python
# tests/conftest.py
import os
from typing import Dict, Any

# Set required env vars before any config.py import occurs
os.environ.setdefault("SONAR_URL", "http://localhost:9000")
os.environ.setdefault("SONAR_TOKEN", "test-token")
os.environ.setdefault("PGVECTOR_DSN", "postgresql://test:test@localhost/test")
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("GITHUB_REPO", "test/repo")
os.environ.setdefault("REPO_LOCAL_PATH", "/tmp/test-repo")
os.environ.setdefault("LLM_MODEL", "claude-sonnet-4-6")

from state import AgentState


def make_state(**kwargs) -> AgentState:
    defaults: Dict[str, Any] = {
        "project_key": "test:project",
        "branch": "main",
        "issues": [],
        "fixes": [],
        "validation_result": None,
        "round_number": 0,
        "max_rounds": 3,
        "pr_url": None,
        "messages": [],
    }
    defaults.update(kwargs)
    return defaults  # type: ignore
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_issue_reader.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (agent not written yet).

- [ ] **Step 4: Write agents/issue_reader/tools.py**

```python
# agents/issue_reader/tools.py
import os
from typing import List, Dict
from mcp.sonarqube_mcp import SonarQubeClient

_client: SonarQubeClient = None


def get_client() -> SonarQubeClient:
    global _client
    if _client is None:
        _client = SonarQubeClient()
    return _client


def fetch_raw_issues(project_key: str, branch: str) -> List[Dict]:
    return get_client().get_issues(project_key, branch)


def fetch_rule_details(rule_key: str) -> Dict:
    return get_client().get_rule(rule_key)


def fetch_source_snippet(component_key: str, line: int, context: int = 3) -> str:
    return get_client().get_source(
        component_key,
        from_line=max(1, line - context),
        to_line=line + context,
    )
```

- [ ] **Step 5: Write agents/issue_reader/agent.py**

```python
# agents/issue_reader/agent.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from state import AgentState, Issue
from agents.issue_reader.tools import (
    fetch_raw_issues, fetch_rule_details, fetch_source_snippet
)


class IssueReaderState(TypedDict):
    project_key: str
    branch: str
    raw_issues: List[Dict]
    rule_cache: Dict[str, Dict]
    issues: List[Issue]


def fetch_issues_node(state: IssueReaderState) -> dict:
    raw = fetch_raw_issues(state["project_key"], state["branch"])
    return {"raw_issues": raw}


def fetch_rule_details_node(state: IssueReaderState) -> dict:
    unique_rules = {i["rule"] for i in state["raw_issues"]}
    cache = {}
    for rule_key in unique_rules:
        cache[rule_key] = fetch_rule_details(rule_key)
    return {"rule_cache": cache}


def fetch_source_node(state: IssueReaderState) -> dict:
    issues: List[Issue] = []
    for raw in state["raw_issues"]:
        rule = state["rule_cache"].get(raw["rule"], {})
        line = raw.get("textRange", {}).get("startLine", 1)
        snippet = fetch_source_snippet(raw["component"], line)
        issues.append(Issue(
            issue_key=raw["key"],
            rule_id=raw["rule"],
            rule_description=rule.get("name", ""),
            remediation_guidance=rule.get("remFnBaseEffort", ""),
            severity=raw.get("severity", ""),
            file_path=raw["component"].split(":")[-1],
            line_start=line,
            line_end=raw.get("textRange", {}).get("endLine", line),
            code_snippet=snippet,
        ))
    return {"issues": issues}


def _build_graph():
    builder = StateGraph(IssueReaderState)
    builder.add_node("fetch_issues", fetch_issues_node)
    builder.add_node("fetch_rules", fetch_rule_details_node)
    builder.add_node("fetch_source", fetch_source_node)
    builder.add_edge(START, "fetch_issues")
    builder.add_edge("fetch_issues", "fetch_rules")
    builder.add_edge("fetch_rules", "fetch_source")
    builder.add_edge("fetch_source", END)
    return builder.compile()


_graph = _build_graph()


def issue_reader_node(state: AgentState) -> dict:
    result = _graph.invoke({
        "project_key": state["project_key"],
        "branch": state["branch"],
        "raw_issues": [],
        "rule_cache": {},
        "issues": [],
    })
    return {"issues": result["issues"]}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_issue_reader.py -v
```

Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add agents/issue_reader/ tests/test_issue_reader.py tests/conftest.py
git commit -m "feat: add IssueReaderAgent subgraph with SonarQube API tools"
```

---

## Task 9: Indentation-Preserving Patch Logic (TDD)

**Files:**
- Create: `agents/remediation/patch.py`
- Create: `tests/test_patch.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_patch.py
import pytest
from agents.remediation.patch import (
    detect_base_indent,
    normalize_indent,
    reapply_indent,
    apply_patch,
    create_unified_diff,
)


def test_detect_base_indent_4spaces():
    code = "    public void method() {\n        String s = \"hello\";\n    }"
    assert detect_base_indent(code) == "    "


def test_detect_base_indent_8spaces():
    code = "        String status = \"PENDING\";"
    assert detect_base_indent(code) == "        "


def test_detect_base_indent_tabs():
    code = "\t\tString s = \"hello\";"
    assert detect_base_indent(code) == "\t\t"


def test_detect_base_indent_no_indent():
    code = "public void method() {}"
    assert detect_base_indent(code) == ""


def test_normalize_removes_base_indent():
    code = "    line1\n    line2\n    line3"
    result = normalize_indent(code)
    assert result == "line1\nline2\nline3"


def test_normalize_preserves_relative_indent():
    code = "    public void m() {\n        return 1;\n    }"
    result = normalize_indent(code)
    assert result == "public void m() {\n    return 1;\n}"


def test_reapply_adds_indent():
    code = "public void m() {\n    return 1;\n}"
    result = reapply_indent(code, "    ")
    assert result == "    public void m() {\n        return 1;\n    }"


def test_reapply_with_tabs():
    code = "public void m() {\n    return 1;\n}"
    result = reapply_indent(code, "\t")
    assert result.startswith("\t")
    assert "\t    return 1;" in result


def test_apply_patch_replaces_lines():
    file_content = (
        "public class Foo {\n"
        "    public void bar() {\n"
        "        String x = \"dup\";\n"
        "        String y = \"dup\";\n"
        "    }\n"
        "}"
    )
    fixed_block = 'public void bar() {\n    private static final String DUP = "dup";\n    String x = DUP;\n    String y = DUP;\n}'
    result = apply_patch(file_content, line_start=2, line_end=5, fixed_block=fixed_block)
    assert "DUP" in result
    assert "    private static final" in result  # indentation preserved
    assert "public class Foo {" in result         # surrounding code intact


def test_create_unified_diff_produces_diff():
    original = "public class Foo {\n    String x = \"dup\";\n}"
    fixed_block = "public class Foo {\n    private static final String DUP = \"dup\";\n    String x = DUP;\n}"
    diff = create_unified_diff(original, fixed_block, "Foo.java", line_start=1, line_end=3)
    assert "---" in diff
    assert "+++" in diff
    assert "-    String x" in diff
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_patch.py -v
```

Expected: `ImportError` — patch.py does not exist yet.

- [ ] **Step 3: Write agents/remediation/patch.py**

```python
# agents/remediation/patch.py
import difflib
from typing import Tuple


def detect_base_indent(code: str) -> str:
    for line in code.split("\n"):
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


def normalize_indent(code: str) -> str:
    base = detect_base_indent(code)
    if not base:
        return code
    result = []
    for line in code.split("\n"):
        if line.startswith(base):
            result.append(line[len(base):])
        elif not line.strip():
            result.append("")
        else:
            result.append(line)
    return "\n".join(result)


def reapply_indent(code: str, base_indent: str) -> str:
    if not base_indent:
        return code
    result = []
    for line in code.split("\n"):
        if line.strip():
            result.append(base_indent + line)
        else:
            result.append(line)
    return "\n".join(result)


def apply_patch(file_content: str, line_start: int, line_end: int,
                fixed_block: str) -> str:
    """Replace lines [line_start, line_end] (1-indexed, inclusive) with fixed_block.

    Preserves the indentation style of the original block.
    """
    original_lines = file_content.split("\n")
    original_block = "\n".join(original_lines[line_start - 1: line_end])
    base_indent = detect_base_indent(original_block)

    normalized = normalize_indent(fixed_block)
    reindented = reapply_indent(normalized, base_indent)

    new_lines = (
        original_lines[: line_start - 1]
        + reindented.split("\n")
        + original_lines[line_end:]
    )
    return "\n".join(new_lines)


def create_unified_diff(original_content: str, new_content: str,
                        file_path: str, line_start: int, line_end: int) -> str:
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_patch.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/remediation/patch.py tests/test_patch.py
git commit -m "feat: add indentation-preserving patch logic with full test coverage"
```

---

## Task 10: RemediationAgent Subgraph

**Files:**
- Create: `agents/remediation/tools.py`
- Create: `agents/remediation/agent.py`
- Create: `tests/test_remediation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_remediation.py
from unittest.mock import patch, MagicMock
from agents.remediation.agent import remediation_node
from state import Issue
from tests.conftest import make_state


def make_issue(**kwargs) -> Issue:
    defaults = Issue(
        issue_key="AX123",
        rule_id="java:S1192",
        rule_description="String literals should not be duplicated",
        remediation_guidance="Extract into a constant",
        severity="MAJOR",
        file_path="src/main/java/PaymentService.java",
        line_start=5,
        line_end=6,
        code_snippet='        String s1 = "PENDING";\n        String s2 = "PENDING";',
    )
    defaults.update(kwargs)
    return defaults


@patch("agents.remediation.tools.RAGRetriever")
@patch("agents.remediation.tools.call_llm")
@patch("builtins.open")
def test_remediation_node_produces_fix(mock_open, mock_llm, mock_rag_cls):
    mock_rag = MagicMock()
    mock_rag_cls.return_value = mock_rag
    mock_rag.search.return_value = [
        {"rule_key": "java:S1192", "description": "...", "remediation": "..."}
    ]
    mock_llm.return_value = (
        'private static final String PENDING = "PENDING";\n'
        "        String s1 = PENDING;\n"
        "        String s2 = PENDING;"
    )
    file_content = (
        "public class PaymentService {\n"
        "    public void pay() {\n"
        '        String s1 = "PENDING";\n'
        '        String s2 = "PENDING";\n'
        "    }\n"
        "}"
    )
    mock_open.return_value.__enter__.return_value.read.return_value = file_content

    issue = make_issue()
    state = make_state(issues=[issue])
    result = remediation_node(state)

    assert len(result["fixes"]) == 1
    fix = result["fixes"][0]
    assert fix["issue_key"] == "AX123"
    assert "PENDING" in fix["fixed_snippet"]
    assert fix["unified_diff"] != ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_remediation.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write agents/remediation/tools.py**

```python
# agents/remediation/tools.py
import os
from typing import List, Dict
from litellm import completion
from rag.retriever import RAGRetriever
from config import LLM_MODEL, REPO_LOCAL_PATH

_retriever: RAGRetriever = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def rag_retrieve(rule_id: str, code_snippet: str, top_k: int = 3) -> List[Dict]:
    query = f"{rule_id}: {code_snippet[:200]}"
    return get_retriever().search(query, top_k=top_k)


def read_file(relative_path: str) -> str:
    full_path = os.path.join(REPO_LOCAL_PATH, relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(relative_path: str, content: str) -> None:
    full_path = os.path.join(REPO_LOCAL_PATH, relative_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def call_llm(rule_id: str, rule_description: str, remediation_guidance: str,
             rag_context: str, file_path: str,
             line_start: int, line_end: int, code_snippet: str) -> str:
    system_prompt = (
        "You are a Java code fixer specializing in SonarQube rule remediation. "
        "You MUST preserve the exact indentation style of the original code. "
        "Return ONLY the fixed code block — no explanations, no markdown fences."
    )
    user_prompt = (
        f"Rule: {rule_id} — {rule_description}\n"
        f"Remediation guidance: {remediation_guidance}\n\n"
        f"Similar rule examples from knowledge base:\n{rag_context}\n\n"
        f"Original code (lines {line_start}–{line_end} of {file_path}):\n"
        f"{code_snippet}\n\n"
        "Fix the above code to resolve the SonarQube rule violation."
    )
    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.1,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Write agents/remediation/agent.py**

```python
# agents/remediation/agent.py
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from state import AgentState, Issue, Fix, ValidationResult
from agents.remediation.tools import rag_retrieve, read_file, write_file, call_llm
from agents.remediation.patch import apply_patch, create_unified_diff


class RemediationState(TypedDict):
    issues_to_fix: List[Issue]
    rag_contexts: Dict[str, str]   # issue_key → rag context string
    fixes: List[Fix]


def rag_retrieve_node(state: RemediationState) -> dict:
    rag_contexts: Dict[str, str] = {}
    for issue in state["issues_to_fix"]:
        docs = rag_retrieve(issue["rule_id"], issue["code_snippet"])
        rag_contexts[issue["issue_key"]] = "\n---\n".join(
            f"{d['rule_key']}: {d['description']} Remediation: {d['remediation']}"
            for d in docs
        )
    return {"rag_contexts": rag_contexts}


def llm_fix_node(state: RemediationState) -> dict:
    fixes: List[Fix] = []
    for issue in state["issues_to_fix"]:
        fixed_snippet = call_llm(
            rule_id=issue["rule_id"],
            rule_description=issue["rule_description"],
            remediation_guidance=issue["remediation_guidance"],
            rag_context=state["rag_contexts"].get(issue["issue_key"], ""),
            file_path=issue["file_path"],
            line_start=issue["line_start"],
            line_end=issue["line_end"],
            code_snippet=issue["code_snippet"],
        )
        file_content = read_file(issue["file_path"])
        new_content = apply_patch(
            file_content,
            line_start=issue["line_start"],
            line_end=issue["line_end"],
            fixed_block=fixed_snippet,
        )
        diff = create_unified_diff(
            file_content, new_content, issue["file_path"],
            issue["line_start"], issue["line_end"]
        )
        write_file(issue["file_path"], new_content)
        fixes.append(Fix(
            issue_key=issue["issue_key"],
            file_path=issue["file_path"],
            unified_diff=diff,
            original_snippet=issue["code_snippet"],
            fixed_snippet=fixed_snippet,
        ))
    return {"fixes": fixes}


def _build_graph():
    builder = StateGraph(RemediationState)
    builder.add_node("rag_retrieve", rag_retrieve_node)
    builder.add_node("llm_fix", llm_fix_node)
    builder.add_edge(START, "rag_retrieve")
    builder.add_edge("rag_retrieve", "llm_fix")
    builder.add_edge("llm_fix", END)
    return builder.compile()


_graph = _build_graph()


def remediation_node(state: AgentState) -> dict:
    vr: Optional[ValidationResult] = state.get("validation_result")
    issues_to_fix = vr["remaining_issues"] if vr else state["issues"]
    result = _graph.invoke({"issues_to_fix": issues_to_fix, "rag_contexts": {}, "fixes": []})
    return {
        "fixes": result["fixes"],
        "validation_result": None,   # reset so validator runs fresh
        "round_number": state["round_number"] + 1,
    }
```

- [ ] **Step 5: Add config.py**

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

SONAR_URL: str = os.environ["SONAR_URL"]
SONAR_TOKEN: str = os.environ["SONAR_TOKEN"]
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
PGVECTOR_DSN: str = os.environ["PGVECTOR_DSN"]
GITHUB_TOKEN: str = os.environ["GITHUB_TOKEN"]
GITHUB_REPO: str = os.environ["GITHUB_REPO"]
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "3"))
REPO_LOCAL_PATH: str = os.environ["REPO_LOCAL_PATH"]
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_remediation.py -v
```

Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add agents/remediation/ tests/test_remediation.py config.py
git commit -m "feat: add RemediationAgent subgraph with RAG + LiteLLM code fix"
```

---

## Task 11: ValidationAgent Subgraph

**Files:**
- Create: `agents/validation/tools.py`
- Create: `agents/validation/agent.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_validation.py
from unittest.mock import patch, MagicMock
from agents.validation.agent import validation_node
from state import Fix
from tests.conftest import make_state


def make_fix(**kwargs) -> Fix:
    defaults = Fix(
        issue_key="AX123",
        file_path="src/main/java/PaymentService.java",
        unified_diff="--- a/PaymentService.java\n+++ b/PaymentService.java\n",
        original_snippet='String s = "PENDING";',
        fixed_snippet="String s = PENDING_STATUS;",
    )
    defaults.update(kwargs)
    return defaults


@patch("mcp.sonarqube_mcp.SonarQubeClient.trigger_scan")
@patch("mcp.sonarqube_mcp.SonarQubeClient.poll_task")
@patch("mcp.sonarqube_mcp.SonarQubeClient.check_issues_resolved")
def test_validation_node_all_resolved(mock_check, mock_poll, mock_scan):
    mock_scan.return_value = "task-id-123"
    mock_poll.return_value = {"status": "SUCCESS", "id": "task-id-123"}
    mock_check.return_value = {
        "resolved": ["AX123"],
        "remaining_raw": [],
    }
    from state import Issue
    issue = Issue(
        issue_key="AX123", rule_id="java:S1192",
        rule_description="", remediation_guidance="",
        severity="MAJOR", file_path="src/main/java/PaymentService.java",
        line_start=5, line_end=5, code_snippet="",
    )
    state = make_state(issues=[issue], fixes=[make_fix()], round_number=1)
    result = validation_node(state)

    vr = result["validation_result"]
    assert vr["all_resolved"] is True
    assert vr["resolved_issues"] == ["AX123"]
    assert vr["remaining_issues"] == []


@patch("mcp.sonarqube_mcp.SonarQubeClient.trigger_scan")
@patch("mcp.sonarqube_mcp.SonarQubeClient.poll_task")
@patch("mcp.sonarqube_mcp.SonarQubeClient.check_issues_resolved")
def test_validation_node_remaining_issues(mock_check, mock_poll, mock_scan):
    mock_scan.return_value = "task-id-456"
    mock_poll.return_value = {"status": "SUCCESS"}
    from state import Issue
    remaining_issue = Issue(
        issue_key="AX123", rule_id="java:S1192",
        rule_description="", remediation_guidance="",
        severity="MAJOR", file_path="src/main/java/PaymentService.java",
        line_start=5, line_end=5, code_snippet="",
    )
    mock_check.return_value = {
        "resolved": [],
        "remaining_raw": [remaining_issue],
    }
    state = make_state(issues=[remaining_issue], fixes=[make_fix()], round_number=1)
    result = validation_node(state)

    vr = result["validation_result"]
    assert vr["all_resolved"] is False
    assert len(vr["remaining_issues"]) == 1
```

- [ ] **Step 2: Write agents/validation/tools.py**

```python
# agents/validation/tools.py
import os
from typing import List, Dict
from mcp.sonarqube_mcp import SonarQubeClient
from config import SONAR_URL, SONAR_TOKEN, REPO_LOCAL_PATH

_client: SonarQubeClient = None


def get_client() -> SonarQubeClient:
    global _client
    if _client is None:
        _client = SonarQubeClient()
    return _client


def run_scan(project_key: str) -> str:
    return get_client().trigger_scan(project_key, SONAR_URL, REPO_LOCAL_PATH)


def wait_for_scan(task_id: str) -> Dict:
    return get_client().poll_task(task_id)


def check_resolved(project_key: str, branch: str,
                   issue_keys: List[str]) -> Dict:
    return get_client().check_issues_resolved(project_key, branch, issue_keys)
```

- [ ] **Step 3: Write agents/validation/agent.py**

```python
# agents/validation/agent.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from state import AgentState, Issue, Fix, ValidationResult
from agents.validation.tools import run_scan, wait_for_scan, check_resolved


class ValidationState(TypedDict):
    project_key: str
    branch: str
    issue_keys: List[str]
    original_issues: List[Issue]
    task_id: str
    validation_result: ValidationResult


def trigger_scan_node(state: ValidationState) -> dict:
    task_id = run_scan(state["project_key"])
    return {"task_id": task_id}


def poll_task_node(state: ValidationState) -> dict:
    wait_for_scan(state["task_id"])
    return {}


def check_issues_node(state: ValidationState) -> dict:
    result = check_resolved(
        state["project_key"],
        state["branch"],
        state["issue_keys"],
    )
    remaining_issues = _rebuild_issue_objects(
        result["remaining_raw"], state["original_issues"]
    )
    vr = ValidationResult(
        resolved_issues=result["resolved"],
        remaining_issues=remaining_issues,
        all_resolved=len(result["remaining_raw"]) == 0,
        round_number=0,
    )
    return {"validation_result": vr}


def _rebuild_issue_objects(remaining_raw: List[dict],
                           original_issues: List[Issue]) -> List[Issue]:
    original_by_key = {i["issue_key"]: i for i in original_issues}
    result = []
    for raw in remaining_raw:
        key = raw.get("issue_key") or raw.get("key", "")
        if key in original_by_key:
            result.append(original_by_key[key])
    return result


def _build_graph():
    builder = StateGraph(ValidationState)
    builder.add_node("trigger_scan", trigger_scan_node)
    builder.add_node("poll_task", poll_task_node)
    builder.add_node("check_issues", check_issues_node)
    builder.add_edge(START, "trigger_scan")
    builder.add_edge("trigger_scan", "poll_task")
    builder.add_edge("poll_task", "check_issues")
    builder.add_edge("check_issues", END)
    return builder.compile()


_graph = _build_graph()


def validation_node(state: AgentState) -> dict:
    issue_keys = [i["issue_key"] for i in state["issues"]]
    result = _graph.invoke({
        "project_key": state["project_key"],
        "branch": state["branch"],
        "issue_keys": issue_keys,
        "original_issues": state["issues"],
        "task_id": "",
        "validation_result": None,
    })
    vr = result["validation_result"]
    vr["round_number"] = state["round_number"]
    return {"validation_result": vr}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_validation.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/validation/ tests/test_validation.py
git commit -m "feat: add ValidationAgent subgraph with sonar-scanner + polling"
```

---

## Task 12: GitHubAgent Subgraph

**Files:**
- Create: `agents/github/tools.py`
- Create: `agents/github/agent.py`

- [ ] **Step 1: Write agents/github/tools.py**

```python
# agents/github/tools.py
import os
import subprocess
from datetime import datetime
from typing import List
from github import Github
from config import GITHUB_TOKEN, GITHUB_REPO, REPO_LOCAL_PATH


def git(cmd: List[str]) -> str:
    result = subprocess.run(
        ["git"] + cmd,
        cwd=REPO_LOCAL_PATH,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def create_branch(project_key: str) -> str:
    safe_key = project_key.replace(":", "-").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    branch_name = f"fix/sonarqube-{safe_key}-{timestamp}"
    git(["checkout", "-b", branch_name])
    return branch_name


def commit_files(file_paths: List[str], rule_id: str, description: str) -> None:
    for fp in file_paths:
        git(["add", fp])
    short_desc = description[:72] if len(description) > 72 else description
    git(["commit", "-m", f"fix({rule_id}): {short_desc}"])


def push_branch(branch_name: str) -> None:
    git(["push", "--set-upstream", "origin", branch_name])


def create_pr(branch_name: str, base_branch: str, pr_body: str,
              n_fixed: int, n_rounds: int) -> str:
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    pr = repo.create_pull(
        title=f"fix: auto-remediate {n_fixed} SonarQube issues ({n_rounds} round(s))",
        body=pr_body,
        head=branch_name,
        base=base_branch,
    )
    return pr.html_url
```

- [ ] **Step 2: Write agents/github/agent.py**

```python
# agents/github/agent.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from state import AgentState, Fix, ValidationResult
from agents.github.tools import create_branch, commit_files, push_branch, create_pr


class GitHubState(TypedDict):
    project_key: str
    branch: str
    fixes: List[Fix]
    validation_result: ValidationResult
    max_rounds: int
    branch_name: str
    pr_url: str


def _build_pr_body(fixes: List[Fix], vr: ValidationResult) -> str:
    rows = "\n".join(
        f"| {f['issue_key']} | `{f['file_path'].split('/')[-1]}` | — |"
        for f in fixes
    )
    resolved_count = len(vr["resolved_issues"])
    remaining_count = len(vr["remaining_issues"])
    scan_line = (
        f"✅ Re-scan passed: {resolved_count} resolved, 0 remaining"
        if vr["all_resolved"]
        else f"⚠️ Re-scan: {resolved_count} resolved, {remaining_count} remaining (max rounds reached)"
    )
    return (
        f"## Auto-fix: SonarQube Issues\n\n"
        f"### Issues Fixed\n"
        f"| Issue Key | File | Rule |\n"
        f"|-----------|------|------|\n"
        f"{rows}\n\n"
        f"### Validation\n"
        f"{scan_line} after round {vr['round_number']}\n\n"
        f"---\n*Generated by auto-sonarqube-reports-fix agent*"
    )


def create_branch_node(state: GitHubState) -> dict:
    branch_name = create_branch(state["project_key"])
    return {"branch_name": branch_name}


def commit_files_node(state: GitHubState) -> dict:
    unique_files = list({f["file_path"] for f in state["fixes"]})
    rule_id = state["fixes"][0]["issue_key"] if state["fixes"] else "sonarqube"
    commit_files(unique_files, rule_id, f"resolve {len(state['fixes'])} SonarQube issues")
    return {}


def create_pr_node(state: GitHubState) -> dict:
    push_branch(state["branch_name"])
    pr_body = _build_pr_body(state["fixes"], state["validation_result"])
    url = create_pr(
        branch_name=state["branch_name"],
        base_branch=state["branch"],
        pr_body=pr_body,
        n_fixed=len(state["fixes"]),
        n_rounds=state["validation_result"]["round_number"],
    )
    return {"pr_url": url}


def _build_graph():
    builder = StateGraph(GitHubState)
    builder.add_node("create_branch", create_branch_node)
    builder.add_node("commit_files", commit_files_node)
    builder.add_node("create_pr", create_pr_node)
    builder.add_edge(START, "create_branch")
    builder.add_edge("create_branch", "commit_files")
    builder.add_edge("commit_files", "create_pr")
    builder.add_edge("create_pr", END)
    return builder.compile()


_graph = _build_graph()


def github_node(state: AgentState) -> dict:
    result = _graph.invoke({
        "project_key": state["project_key"],
        "branch": state["branch"],
        "fixes": state["fixes"],
        "validation_result": state["validation_result"],
        "max_rounds": state["max_rounds"],
        "branch_name": "",
        "pr_url": "",
    })
    return {"pr_url": result["pr_url"]}
```

- [ ] **Step 3: Commit**

```bash
git add agents/github/
git commit -m "feat: add GitHubAgent subgraph for branch + PR creation"
```

---

## Task 13: Supervisor Orchestrator

**Files:**
- Create: `orchestrator/supervisor.py`
- Create: `tests/test_supervisor.py`

- [ ] **Step 1: Write failing routing tests**

```python
# tests/test_supervisor.py
from orchestrator.supervisor import route
from tests.conftest import make_state
from state import Issue, Fix, ValidationResult


def _issue() -> Issue:
    return Issue(
        issue_key="AX1", rule_id="java:S1192",
        rule_description="", remediation_guidance="",
        severity="MAJOR", file_path="Foo.java",
        line_start=1, line_end=1, code_snippet="",
    )


def _fix() -> Fix:
    return Fix(
        issue_key="AX1", file_path="Foo.java",
        unified_diff="---\n+++\n", original_snippet="", fixed_snippet="",
    )


def test_route_to_issue_reader_when_no_issues():
    assert route(make_state()) == "issue_reader"


def test_route_to_remediator_when_issues_no_fixes():
    assert route(make_state(issues=[_issue()])) == "remediator"


def test_route_to_validator_when_fixes_no_validation():
    assert route(make_state(issues=[_issue()], fixes=[_fix()])) == "validator"


def test_route_to_remediator_when_remaining_issues_and_rounds_left():
    vr = ValidationResult(
        resolved_issues=[], remaining_issues=[_issue()],
        all_resolved=False, round_number=1,
    )
    state = make_state(issues=[_issue()], fixes=[_fix()],
                       validation_result=vr, round_number=1, max_rounds=3)
    assert route(state) == "remediator"


def test_route_to_github_when_all_resolved():
    vr = ValidationResult(
        resolved_issues=["AX1"], remaining_issues=[],
        all_resolved=True, round_number=1,
    )
    state = make_state(issues=[_issue()], fixes=[_fix()],
                       validation_result=vr, round_number=1)
    assert route(state) == "github_agent"


def test_route_to_github_when_max_rounds_reached():
    vr = ValidationResult(
        resolved_issues=[], remaining_issues=[_issue()],
        all_resolved=False, round_number=3,
    )
    state = make_state(issues=[_issue()], fixes=[_fix()],
                       validation_result=vr, round_number=3, max_rounds=3)
    assert route(state) == "github_agent"
```

- [ ] **Step 2: Run test to verify they fail**

```bash
pytest tests/test_supervisor.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write orchestrator/supervisor.py**

```python
# orchestrator/supervisor.py
from typing import Literal
from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.issue_reader.agent import issue_reader_node
from agents.remediation.agent import remediation_node
from agents.validation.agent import validation_node
from agents.github.agent import github_node

Route = Literal["issue_reader", "remediator", "validator", "github_agent"]


def route(state: AgentState) -> Route:
    if not state["issues"]:
        return "issue_reader"
    if not state["fixes"]:
        return "remediator"
    if state["validation_result"] is None:
        return "validator"
    vr = state["validation_result"]
    if vr["remaining_issues"] and state["round_number"] < state["max_rounds"]:
        return "remediator"
    return "github_agent"


def _noop(state: AgentState) -> dict:
    return {}


def build_supervisor(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("router", _noop)
    builder.add_node("issue_reader", issue_reader_node)
    builder.add_node("remediator", remediation_node)
    builder.add_node("validator", validation_node)
    builder.add_node("github_agent", github_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", route,
        {
            "issue_reader": "issue_reader",
            "remediator": "remediator",
            "validator": "validator",
            "github_agent": "github_agent",
        },
    )
    builder.add_edge("issue_reader", "router")
    builder.add_edge("remediator", "router")
    builder.add_edge("validator", "router")
    builder.add_edge("github_agent", END)

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run routing tests**

```bash
pytest tests/test_supervisor.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/supervisor.py tests/test_supervisor.py
git commit -m "feat: add Supervisor graph with conditional routing across 4 subagents"
```

---

## Task 14: CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
# main.py
"""CLI entry point for the SonarQube AI Auto-Fix Agent."""
import argparse
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()

from db.sqlite import checkpointer
from orchestrator.supervisor import build_supervisor
from state import AgentState

supervisor = build_supervisor(checkpointer=checkpointer)


def run(project_key: str, branch: str, max_rounds: int,
        github_repo: str, thread_id: str) -> str:
    import os
    os.environ.setdefault("GITHUB_REPO", github_repo)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(
        project_key=project_key,
        branch=branch,
        issues=[],
        fixes=[],
        validation_result=None,
        round_number=0,
        max_rounds=max_rounds,
        pr_url=None,
        messages=[],
    )

    print(f"[agent] Starting run — thread_id={thread_id}")
    final = supervisor.invoke(initial_state, config=config)
    return final.get("pr_url", "")


def resume(thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[agent] Resuming run — thread_id={thread_id}")
    final = supervisor.invoke({}, config=config)
    return final.get("pr_url", "")


def main():
    parser = argparse.ArgumentParser(
        description="SonarQube AI Auto-Fix Agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new fix run")
    run_parser.add_argument("--project", required=True, help="SonarQube project key")
    run_parser.add_argument("--branch", default="main")
    run_parser.add_argument("--max-rounds", type=int, default=3)
    run_parser.add_argument("--github-repo", required=True,
                            help="GitHub repo e.g. myorg/payment-service")
    run_parser.add_argument("--thread-id", default=None,
                            help="Optional: reuse a specific thread ID")

    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted run")
    resume_parser.add_argument("--thread-id", required=True)

    args = parser.parse_args()

    if args.command == "run":
        thread_id = args.thread_id or str(uuid.uuid4())
        print(f"[agent] thread_id: {thread_id}  (use --thread-id {thread_id} to resume)")
        pr_url = run(
            project_key=args.project,
            branch=args.branch,
            max_rounds=args.max_rounds,
            github_repo=args.github_repo,
            thread_id=thread_id,
        )
        print(f"[agent] Done. PR: {pr_url}")
        sys.exit(0)

    if args.command == "resume":
        pr_url = resume(args.thread_id)
        print(f"[agent] Resumed. PR: {pr_url}")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

```bash
python main.py --help
python main.py run --help
```

Expected: help text with `--project`, `--branch`, `--max-rounds`, `--github-repo`.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with run/resume subcommands"
```

---

## Task 15: Full Test Suite + Smoke Check

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: All tests pass. Typical output:
```
tests/test_patch.py::test_detect_base_indent_4spaces PASSED
tests/test_patch.py::test_detect_base_indent_8spaces PASSED
tests/test_patch.py::test_detect_base_indent_tabs PASSED
tests/test_patch.py::test_detect_base_indent_no_indent PASSED
tests/test_patch.py::test_normalize_removes_base_indent PASSED
tests/test_patch.py::test_normalize_preserves_relative_indent PASSED
tests/test_patch.py::test_reapply_adds_indent PASSED
tests/test_patch.py::test_reapply_with_tabs PASSED
tests/test_patch.py::test_apply_patch_replaces_lines PASSED
tests/test_patch.py::test_create_unified_diff_produces_diff PASSED
tests/test_supervisor.py::test_route_to_issue_reader_when_no_issues PASSED
tests/test_supervisor.py::test_route_to_remediator_when_issues_no_fixes PASSED
tests/test_supervisor.py::test_route_to_validator_when_fixes_no_validation PASSED
tests/test_supervisor.py::test_route_to_remediator_when_remaining_issues_and_rounds_left PASSED
tests/test_supervisor.py::test_route_to_github_when_all_resolved PASSED
tests/test_supervisor.py::test_route_to_github_when_max_rounds_reached PASSED
tests/test_issue_reader.py::test_issue_reader_node_maps_api_response PASSED
tests/test_issue_reader.py::test_issue_reader_returns_empty_when_no_issues PASSED
tests/test_remediation.py::test_remediation_node_produces_fix PASSED
tests/test_validation.py::test_validation_node_all_resolved PASSED
tests/test_validation.py::test_validation_node_remaining_issues PASSED
```

- [ ] **Step 2: Verify supervisor graph compiles without errors**

```bash
python -c "
from orchestrator.supervisor import build_supervisor
g = build_supervisor()
print('Supervisor graph nodes:', list(g.nodes))
"
```

Expected output includes: `['router', 'issue_reader', 'remediator', 'validator', 'github_agent']`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: full test suite passing — all 21 tests green"
```

---

## Summary

| Task | What It Builds |
|------|---------------|
| 1 | Project skeleton, docker-compose, deps |
| 2 | Shared TypedDicts (Issue, Fix, ValidationResult, AgentState) |
| 3 | SQLite checkpointer for resumable runs |
| 4 | EmbeddingModel (OpenAI / local) |
| 5 | RAGRetriever with pgvector cosine search |
| 6 | Offline ingest: 600+ Java rules → pgvector |
| 7 | SonarQubeClient REST wrapper |
| 8 | IssueReaderAgent (fetch → rules → source) |
| 9 | Indentation-preserving patch logic with TDD |
| 10 | RemediationAgent (RAG → LiteLLM → patch) |
| 11 | ValidationAgent (scan → poll → check) |
| 12 | GitHubAgent (branch → commit → PR) |
| 13 | Supervisor graph with 6-case routing |
| 14 | CLI with run/resume subcommands |
| 15 | Full test suite (21 tests) |
