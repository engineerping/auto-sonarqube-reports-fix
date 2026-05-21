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
