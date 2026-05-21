# db/sqlite.py
import os
from langgraph.checkpoint.sqlite import SqliteSaver

os.makedirs("runs", exist_ok=True)

checkpointer = SqliteSaver.from_conn_string("runs/agent_runs.db")
