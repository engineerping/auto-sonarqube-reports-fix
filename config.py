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
