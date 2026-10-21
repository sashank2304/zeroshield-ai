import json
from pathlib import Path


class MitreService:
    def __init__(self, root: Path):
        path = root / "model_training" / "outputs" / "Deep_SVDD_MITRE_mapping_for_LLM.json"
        self.records = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.records = {item["attack_class"]: item for item in data.get("mappings", [])}
        except Exception:
            import logging
            logging.exception("Could not load the existing MITRE mapping output")
        self.index = None
        self.index_error = None
        # Reuse the teammate's prebuilt TF-IDF index; never build it per alert.
        try:
            import pickle
            index_path = root / "LLM" / "rag_store" / "mitre_tfidf.pkl"
            if index_path.exists():
                with index_path.open("rb") as handle:
                    self.index = pickle.load(handle)
        except Exception as exc:
            self.index_error = str(exc)

    @property
    def available(self):
        return bool(self.records)

    def insufficient(self):
        return {"status": "insufficient_evidence", "message": "Insufficient evidence for a confident ATT&CK mapping.", "techniques": []}

    def retrieve_behavior(self, analysis):
        if not self.index:
            return {**self.insufficient(), "status": "retrieval_unavailable", "message": "Behavior retrieval unavailable; no techniques fabricated.", "error": self.index_error}
        evidence = analysis.get("top_features", [])[:8]
        # Query is generated from observed values and contribution directions, without labels.
        query = "Network flow behavior: " + "; ".join(
            f"{x['feature']} value {x['value']:.5g}, perturbation contribution {x['contribution']:.5g}"
            for x in evidence
        ) + f"; anomaly score {analysis['anomaly_score']:.6g} against threshold {analysis['threshold']:.6g}."
        store = self.index
        try:
            scores = (store["vectorizer"].transform([query]) @ store["matrix"].T).toarray().ravel()
            results = []
            for index in scores.argsort()[::-1]:
                if scores[index] <= 0: break
                doc = store["documents"][int(index)]
                results.append({"technique_id": doc["id"], "technique_name": doc["name"],
                                "description": doc["text"].split("Description: ", 1)[-1][:1200],
                                "source": doc.get("url"), "similarity": float(scores[index]),
                                "tactics": doc.get("tactics", []), "platforms": doc.get("platforms", [])})
                if len(results) == 5: break
        except Exception as exc:
            return {**self.insufficient(), "status": "retrieval_unavailable", "message": "Behavior retrieval failed; no techniques fabricated.", "error": str(exc), "query": query}
        return {"status": "retrieved" if results else "empty_retrieval", "query": query,
                "method": "existing TF-IDF cosine similarity index", "techniques": results,
                "message": "Retrieved ATT&CK candidates are contextual suggestions, not confirmed classifications." if results else "No relevant ATT&CK documents found."}

    def for_category(self, category, analysis):
        item = self.records.get(category)
        if not item:
            return self.insufficient()
        techniques = item.get("allowed_mitre_candidates", [])
        if not techniques:
            return self.insufficient()
        score_statement = f"Deep SVDD score {analysis['anomaly_score']:.6f} {'exceeded' if analysis['detected'] else 'did not exceed'} threshold {analysis['threshold']:.6f}."
        feature_statement = [f"{x['feature']} perturbation contribution: {x['contribution']:.6f} (prepared-space value {x['value']:.6f})." for x in analysis['top_features'][:5]]
        return {"status": "contextual_candidates", "message": "Candidate context from the existing human-reviewed project mapping; flow evidence does not prove a technique.", "techniques": techniques, "evidence": [f"User-selected dataset category context: {category}.", score_statement, *feature_statement], "xai_features": item.get("top_xai_features", [])}
