import pickle
import sys
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path("rag_store/mitre_tfidf.pkl")

if not INDEX_PATH.exists():
    raise SystemExit("Index not found. Run python build_mitre_rag.py first.")

with INDEX_PATH.open("rb") as f:
    store = pickle.load(f)

documents = store["documents"]
vectorizer = store["vectorizer"]
matrix = store["matrix"]


def retrieve(query, top_k=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1]

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue

        doc = documents[idx]
        results.append((float(scores[idx]), doc))

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Enter a SOC investigation query: ").strip()

    if not query:
        raise SystemExit("Please enter a query.")

    results = retrieve(query)

    print(f"\nQuery: {query}")
    print(f"Relevant techniques found: {len(results)}")

    for rank, (score, doc) in enumerate(results, start=1):
        print(f"\n{'=' * 65}")
        print(f"{rank}. {doc['name']} ({doc['id']})")
        print(f"Relevance score: {score:.4f}")
        print(f"Tactics: {', '.join(doc['tactics']) or 'Not listed'}")
        print(f"Platforms: {', '.join(doc['platforms']) or 'Not listed'}")
        print(f"Description: {doc['text'].split('Description: ', 1)[-1][:700]}")
        print(f"MITRE source: {doc['url']}")
