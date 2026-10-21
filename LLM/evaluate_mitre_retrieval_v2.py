import json
import pickle
from pathlib import Path

ROOT = Path(".")
INDEX_PATH = ROOT / "rag_store" / "mitre_tfidf.pkl"
XAI_PATH = ROOT / "data" / "xai_mitre_rag_result.json"
OUT_PATH = ROOT / "data" / "mitre_retrieval_comparison_v2.json"

with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

with open(XAI_PATH, "r", encoding="utf-8") as f:
    xai_result = json.load(f)

if not isinstance(index, dict):
    raise TypeError("Expected the saved TF-IDF index to be a dictionary.")

vectorizer = index["vectorizer"]
matrix = index["matrix"]
documents = index["documents"]

# Preserve the query used by the existing XAI retrieval run.
def find_query(obj):
    if isinstance(obj, dict):
        for key in ("query", "retrieval_query", "rag_query"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in obj.values():
            found = find_query(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_query(value)
            if found:
                return found
    return None

existing_xai_query = find_query(xai_result)

if not existing_xai_query:
    raise ValueError(
        "Could not find the existing XAI query in the saved JSON. "
        "Inspect data/xai_mitre_rag_result.json before proceeding."
    )

# These queries use behavior concepts, not the ground-truth dataset label.
queries = {
    "direct_flooding_baseline": (
        "network traffic flooding high volume packets "
        "denial of service UDP TCP"
    ),
    "behavior_focused": (
        "adversary sends high volume network packets toward a target "
        "to cause denial of service; network flooding using UDP ICMP "
        "or TCP; botnet traffic and reduced network availability"
    ),
    "protocol_feature_focused": (
        "network protocol characteristics TCP UDP packet header "
        "flags SYN ACK FIN RST connection behavior"
    ),
    "existing_xai_query": existing_xai_query,
}

def retrieve(query, top_k=5):
    q_vector = vectorizer.transform([query])
    scores = (matrix @ q_vector.T).toarray().ravel()
    ranked = scores.argsort()[::-1][:top_k]

    results = []
    for rank, i in enumerate(ranked, start=1):
        doc = documents[i]
        results.append({
            "rank": rank,
            "id": doc.get("id", "unknown"),
            "name": doc.get("name", "unknown"),
            "similarity": round(float(scores[i]), 6),
            "url": doc.get("url", ""),
        })
    return results

comparison = {
    "note": (
        "TF-IDF similarity is text similarity, not attack probability. "
        "Queries do not include the ground-truth dataset label. "
        "Rankings require human review before use in SOC reports."
    ),
    "results": {},
}

for name, query in queries.items():
    comparison["results"][name] = {
        "query": query,
        "top_5": retrieve(query),
    }

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

for name, item in comparison["results"].items():
    print(f"\n=== {name} ===")
    print(f'Query: {item["query"]}')
    for result in item["top_5"]:
        print(
            f'{result["rank"]}. {result["id"]} '
            f'{result["name"]} '
            f'(similarity={result["similarity"]})'
        )

print(f"\nSaved results to: {OUT_PATH}")
