import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE = Path("data/mitre_enterprise_techniques.json")
OUT_DIR = Path("rag_store")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with SOURCE.open("r", encoding="utf-8") as f:
    techniques = json.load(f)

documents = []

for technique_id, item in techniques.items():
    name = item.get("name", "")
    description = item.get("description", "")
    tactics = item.get("tactics", [])
    platforms = item.get("platforms", [])

    text = (
        f"Technique ID: {technique_id}\n"
        f"Technique name: {name}\n"
        f"Tactics: {', '.join(tactics)}\n"
        f"Platforms: {', '.join(platforms)}\n"
        f"Description: {description}"
    )

    documents.append({
        "id": technique_id,
        "name": name,
        "tactics": tactics,
        "platforms": platforms,
        "url": item.get("url", ""),
        "text": text,
    })

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=100000,
)

matrix = vectorizer.fit_transform([doc["text"] for doc in documents])

with (OUT_DIR / "mitre_tfidf.pkl").open("wb") as f:
    pickle.dump({
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }, f)

print(f"Loaded techniques: {len(documents)}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Matrix shape: {matrix.shape}")
print(f"Saved index: {OUT_DIR / 'mitre_tfidf.pkl'}")
