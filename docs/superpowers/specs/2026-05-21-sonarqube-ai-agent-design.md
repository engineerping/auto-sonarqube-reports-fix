# SonarQube AI Auto-Fix Agent — Design Spec

**Date:** 2026-05-21  
**Author:** engineerping  
**Status:** Approved

---

## 1. Problem Statement

Our bank deploys SonarQube Server on a private cloud. Every GitHub push triggers a SonarQube scan in CI. The resulting issue backlog grows faster than engineers can manually fix it. This project builds a multi-agent AI system that reads SonarQube issues, auto-generates code fixes, validates them with a re-scan, and raises a GitHub PR — with zero manual intervention after triggering.

Secondary goal: demonstrate a full-breadth AI agent skill set on a resume (LangGraph, MCP, RAG, pgvector, SQLite, LiteLLM).

---

## 2. Scope & Constraints

| Item | Decision |
|------|----------|
| Target language | Java (primary) |
| SonarQube deployment | Self-hosted Server on private cloud |
| LLM | LiteLLM abstraction layer; demo uses `claude-sonnet-4-6`, swappable to Azure OpenAI via env var |
| Trigger | Manual CLI (`python main.py --project ... --branch ...`) |
| RAG content | SonarQube Java rule documentation (descriptions + remediation guidance) |
| Max fix rounds | 3 (configurable) |
| Out of scope | Auto-merge PRs; non-Java languages; SonarQube Cloud |
| Prerequisite | Target repo must be `git clone`d locally before running the agent; the agent reads and patches files on disk |

---

## 3. Architecture

### 3.1 High-Level Overview

```
CLI: python main.py --project <key> --branch <branch>
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  Orchestrator  (LangGraph Supervisor Graph)           │
│  • SqliteSaver checkpointer  (断点续跑)               │
│  • AgentState TypedDict      (共享状态)               │
│  • Conditional routing logic                         │
└──────┬───────────────────────────────────────────────┘
       │ invokes as compiled subgraphs
       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ IssueReader  │──▶│ Remediation  │──▶│  Validation  │──▶│   GitHub     │
│   Agent      │   │    Agent     │   │    Agent     │   │    Agent     │
│  (subgraph)  │   │  (subgraph)  │   │  (subgraph)  │   │  (subgraph)  │
└──────────────┘   └──────┬───────┘   └──────┬───────┘   └──────────────┘
                          │                  │
                   remaining issues ─────────┘
                   (up to max_rounds)
```

### 3.2 Supervisor Routing Logic

```
state.issues is empty?               → route to IssueReaderAgent
state.fixes is empty?                → route to RemediationAgent
state.validation_result is None?     → route to ValidationAgent
remaining_issues > 0 AND round < max → route back to RemediationAgent
all resolved OR round >= max         → route to GitHubAgent → END
```

### 3.3 Shared State

```python
class Issue(TypedDict):
    issue_key: str
    rule_id: str                  # e.g. "java:S1192"
    rule_description: str
    remediation_guidance: str
    severity: str                 # BLOCKER / CRITICAL / MAJOR / MINOR
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str

class Fix(TypedDict):
    issue_key: str
    file_path: str
    unified_diff: str             # unified diff format
    original_snippet: str
    fixed_snippet: str

class ValidationResult(TypedDict):
    resolved_issues: List[str]    # issue_key list
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

---

## 4. SubAgent Designs

### 4.1 IssueReaderAgent

**Purpose:** Query SonarQube Server and return a structured list of all open issues with full rule context.

**Internal graph nodes:**
1. `fetch_issues` — `GET /api/issues/search?projectKeys={key}&branch={branch}&statuses=OPEN`
2. `fetch_rule_details` — `GET /api/rules/show?key={ruleId}` for each unique rule (batched, deduplicated)
3. `fetch_source_code` — `GET /api/sources/raw?key={component}&from={line-5}&to={line+5}` for code context

**Tools (via SonarQube MCP Server — `sonarqube-agent-plugins`):**
- `sonar_list_issues(project_key, branch, severities)`
- `sonar_get_rule(rule_key)` → description + remediation text
- `sonar_get_source(component_key, from_line, to_line)`

**Output:** Populates `state.issues`.

---

### 4.2 RemediationAgent

**Purpose:** For each issue, retrieve relevant rule docs via RAG, then use LiteLLM to generate an indentation-preserving code fix.

**Internal graph nodes:**
1. `rag_retrieve` — query pgvector with `rule_id + code_snippet` as embedding query; returns top-3 similar rule docs as few-shot context
2. `read_file` — read full source file from disk (local workspace clone)
3. `llm_fix` — LiteLLM call with structured prompt
4. `apply_patch` — indentation-preserving patch written back to file

**LiteLLM Prompt Structure:**
```
System: "You are a Java code fixer specializing in SonarQube rule remediation.
         You MUST preserve the exact indentation style of the original code.
         Return ONLY the fixed code block, nothing else."

User:   "Rule: {rule_id} — {rule_description}
         Remediation guidance: {remediation_guidance}
         Similar examples from knowledge base:
         {rag_context}
         
         Original code (lines {start}-{end} of {file_path}):
         {code_snippet}
         
         Fix the above code to resolve the SonarQube rule violation."
```

**Indentation Preservation Strategy:**
1. Before extraction: record `base_indent` (leading whitespace of first line, detect tabs vs spaces)
2. LLM is instructed to output normalized code (4-space indent relative to block start)
3. After LLM response: re-apply `base_indent` offset to every line of the fixed block
4. Generate unified diff against original file; store in `Fix.unified_diff`
5. Only changed lines are written back; surrounding code untouched

**Output:** Populates `state.fixes`.

---

### 4.3 ValidationAgent

**Purpose:** Apply fixes to disk, trigger a SonarQube re-scan, and determine which issues remain.

**Internal graph nodes:**
1. `apply_fixes` — apply all `unified_diff` patches from `state.fixes` to files on disk
2. `trigger_scan` — invoke `sonar-scanner` CLI with project properties
3. `poll_task` — `GET /api/ce/task?id={taskId}` every 10s until `status == SUCCESS`
4. `check_issues` — `GET /api/issues/search` with original issue keys to see resolved status

**Output:** Populates `state.validation_result`. If `remaining_issues` is non-empty, Supervisor routes back to RemediationAgent with only the unresolved issues.

---

### 4.4 GitHubAgent

**Purpose:** Commit all fixed files to a new branch and open a descriptive PR.

**Internal graph nodes:**
1. `create_branch` — `git checkout -b fix/sonarqube-{project_key}-{timestamp}`
2. `commit_files` — `git add {file}` + `git commit -m "fix({rule_id}): {short_description}"` per file
3. `create_pr` — GitHub REST API `POST /repos/{owner}/{repo}/pulls`

**PR body template:**
```markdown
## Auto-fix: SonarQube Issues ({N} issues resolved in {R} rounds)

### Issues Fixed
| Rule ID      | File                    | Severity | Description               |
|-------------|------------------------|----------|--------------------------|
| java:S1192  | PaymentService.java     | MAJOR    | String literal duplicated |

### Validation
✅ SonarQube re-scan: 0 remaining issues after round {R}

---
*Generated by auto-sonarqube-reports-fix agent*
```

**Tools:** `PyGithub` + `subprocess` (git CLI)

**Output:** Populates `state.pr_url`.

---

## 5. RAG Pipeline

### 5.1 Offline Indexing (run once)

```bash
python rag/ingest.py --sonar-url http://sonar.internal --token $SONAR_TOKEN
```

Steps:
1. `GET /api/rules/search?languages=java&ps=500` (paginated, ~600+ Java rules)
2. Compose document per rule: `"{rule_key}: {name}. {description}. Remediation: {remediation_text}"`
3. Embed with `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (local, no external call)
4. Upsert into PostgreSQL/pgvector table `sonar_rules`

### 5.2 pgvector Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sonar_rules (
    id          SERIAL PRIMARY KEY,
    rule_key    VARCHAR(100) UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    remediation TEXT,
    severity    VARCHAR(20),
    embedding   vector(1536)
);

CREATE INDEX ON sonar_rules USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### 5.3 Runtime Query (RemediationAgent)

```python
query_text = f"{issue.rule_id}: {issue.rule_description}. Code: {issue.code_snippet}"
query_embedding = embed(query_text)

results = db.execute("""
    SELECT rule_key, description, remediation
    FROM sonar_rules
    ORDER BY embedding <=> %s
    LIMIT 3
""", [query_embedding])
```

---

## 6. LiteLLM Configuration (Swappable LLM)

```python
# .env (demo default)
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...

# To switch to Azure OpenAI (production):
# LLM_MODEL=azure/gpt-4o
# AZURE_API_KEY=...
# AZURE_API_BASE=https://your-instance.openai.azure.com
# AZURE_API_VERSION=2024-02-01
```

```python
# agents/remediation/tools.py
from litellm import completion
import os

def call_llm(messages: list) -> str:
    response = completion(
        model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
        messages=messages,
        max_tokens=2048,
        temperature=0.1,   # low temp for deterministic code fixes
    )
    return response.choices[0].message.content
```

---

## 7. SQLite Checkpointing

```python
# db/sqlite.py
from langgraph.checkpoint.sqlite import SqliteSaver
import os

os.makedirs("runs", exist_ok=True)
checkpointer = SqliteSaver.from_conn_string("runs/agent_runs.db")
```

Every Supervisor invocation uses a unique `thread_id`. To resume an interrupted run:

```bash
python main.py --resume --thread-id <id>
```

---

## 8. Project Directory Structure

```
auto-sonarqube-reports-fix/
├── main.py                        # CLI entry point (argparse)
├── config.py                      # Env vars + global config
├── state.py                       # AgentState + Issue/Fix/ValidationResult TypedDicts
│
├── orchestrator/
│   └── supervisor.py              # LangGraph Supervisor Graph + routing
│
├── agents/
│   ├── issue_reader/
│   │   ├── agent.py               # IssueReaderAgent subgraph
│   │   └── tools.py               # SonarQube MCP/REST tools
│   ├── remediation/
│   │   ├── agent.py               # RemediationAgent subgraph
│   │   ├── tools.py               # RAG retrieval + file read/write tools
│   │   └── patch.py               # Indentation-preserving patch logic
│   ├── validation/
│   │   ├── agent.py               # ValidationAgent subgraph
│   │   └── tools.py               # sonar-scanner CLI + polling tools
│   └── github/
│       ├── agent.py               # GitHubAgent subgraph
│       └── tools.py               # PyGithub + git CLI tools
│
├── rag/
│   ├── ingest.py                  # Offline: index SonarQube rules → pgvector
│   ├── retriever.py               # RAGRetriever class (query pgvector)
│   └── embeddings.py              # Embedding model wrapper (OpenAI / local)
│
├── db/
│   └── sqlite.py                  # LangGraph SqliteSaver initialisation
│
├── mcp/
│   └── sonarqube_mcp.py           # SonarQube MCP Server client wrapper
│
├── tests/
│   ├── test_issue_reader.py
│   ├── test_remediation.py
│   ├── test_patch.py              # Indentation logic unit tests (critical)
│   └── test_validation.py
│
├── docker-compose.yml             # PostgreSQL + pgvector
├── .env.example                   # Environment variable template
└── requirements.txt
```

---

## 9. Key Dependencies

```
# requirements.txt
langgraph>=0.2
langchain-core>=0.2
litellm>=1.40
anthropic>=0.30
pgvector>=0.2
psycopg2-binary>=2.9
PyGithub>=2.3
openai>=1.30                  # for embeddings (text-embedding-3-small)
sentence-transformers>=3.0    # local embedding alternative
sonarqube-mcp                 # sonarqube-agent-plugins MCP server
python-dotenv>=1.0
```

---

## 10. Usage

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Index SonarQube rules into pgvector (once)
python rag/ingest.py --sonar-url http://sonar.internal --token $SONAR_TOKEN

# 3. Run the agent
python main.py \
  --project com.bank:payment-service \
  --branch main \
  --max-rounds 3 \
  --github-repo myorg/payment-service

# 4. Resume interrupted run
python main.py --resume --thread-id abc-123
```

---

## 11. Resume Value (Interview Talking Points)

| Skill | Demonstrated By |
|-------|----------------|
| LangGraph Supervisor Pattern | Orchestrator routing across 4 compiled subgraphs |
| LangGraph SubGraphs | Each agent is an independently compiled `StateGraph` |
| LangGraph Checkpointing | `SqliteSaver` enabling resumable multi-round runs |
| MCP Integration | SonarQube MCP Server tools in IssueReader + Validation |
| RAG Pipeline | pgvector similarity search enriching LLM fix prompts |
| pgvector | Cosine similarity index over 600+ SonarQube rule embeddings |
| LiteLLM | Single call interface, swap Claude ↔ Azure OpenAI via env var |
| SQLite | Persistent checkpoint store across agent runs |
| Multi-round Agentic Loop | Supervisor retries Remediation → Validation until resolved |
| GitHub Automation | Auto PR creation with structured issue summary |
