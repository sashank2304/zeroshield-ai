import json
import os


def reason(result, family, mitre):
    evidence = {"detected": result["detected"], "anomaly_score": result["anomaly_score"], "threshold": result["threshold"], "top_features": result["top_features"], "attack_family": family, "mitre_candidates": mitre.get("techniques", [])}
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": "You are an evidence-bound SOC analyst. DO NOT invent evidence, anomaly scores, attack labels, features, or MITRE techniques. Use only supplied evidence. Clearly distinguish observation from interpretation and state uncertainty. Return JSON fields: incident_summary, evidence, attack_context, interpretation, recommended_action, limitations."}, {"role": "user", "content": json.dumps(evidence)}])
            return {"mode": "OpenAI LLM", "report": json.loads(response.choices[0].message.content)}
        except Exception:
            import logging
            logging.exception("Optional OpenAI reasoning failed; using labeled fallback")
    state = "exceeded" if result["detected"] else "did not exceed"
    return {"mode": "Demo Fallback Reasoning", "report": {"incident_summary": f"The observed flow score {state} the configured Deep SVDD threshold.", "evidence": [f"Score {result['anomaly_score']:.6f}; threshold {result['threshold']:.6f}."] + [f"{x['feature']} contributed {x['contribution']:.6f}." for x in result["top_features"][:3]], "attack_context": mitre.get("message", "Insufficient evidence for a confident ATT&CK mapping."), "interpretation": "This is a flow-level anomaly result; it does not establish the underlying application or host behavior.", "recommended_action": "Correlate this flow with device, endpoint, and adjacent network telemetry." if result["detected"] else "Continue monitoring and correlate with surrounding flows.", "limitations": "Feature perturbation describes model score sensitivity. Dataset family labels, when present, are context and not inferred by Deep SVDD."}}
