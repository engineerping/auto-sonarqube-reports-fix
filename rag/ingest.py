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
