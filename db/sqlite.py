# db/sqlite.py
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

os.makedirs("runs", exist_ok=True)

_conn = sqlite3.connect("runs/agent_runs.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)
