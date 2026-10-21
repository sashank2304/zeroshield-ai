
"""
ZeroShield AI - Local LLM SOC Assistant
Uses Ollama + llama3.2:3b.
No API key required.

Run:
    python local_llm_soc_assistant.py
"""

import json
import sys
import urllib.request
import urllib.error
import socket
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_REPORT = BASE_DIR / "soc_reports" / "sample_0_soc_report.json"
MITRE_FILE = BASE_DIR / "data" / "mitre_retrieval_comparison_v2.json"

OUTPUT_DIR = BASE_DIR / "soc_reports"
OUTPUT_JSON = OUTPUT_DIR / "sample_0_llm_soc_report.json"
OUTPUT_MD = OUTPUT_DIR / "sample_0_llm_soc_report.md"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

MODEL_NAME = "llama3.2:3b"

# Allow extra time for CPU-based inference.
REQUEST_TIMEOUT = 900

# Keep the response short to reduce generation time.
MAX_OUTPUT_TOKENS = 600


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}\n\n"
            "Check the project folder and make sure the XAI "
            "and MITRE scripts have already generated their outputs."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    def convert_value(value):
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=convert_value
        )


# ============================================================
# OLLAMA CONNECTION
# ============================================================

def check_ollama():
    print("[CHECK] Connecting to Ollama...")

    request = urllib.request.Request(
        OLLAMA_TAGS_URL,
        headers={"Accept": "application/json"}
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

    except (urllib.error.URLError, socket.timeout, TimeoutError) as error:
        raise RuntimeError(
            "Cannot connect to Ollama at localhost:11434.\n"
            "Make sure Ollama is running.\n"
            "Test it with:\n"
            "curl http://localhost:11434/api/tags\n"
            f"\nDetails: {error}"
        ) from error

    models = [
        item.get("name", "")
        for item in data.get("models", [])
    ]

    if MODEL_NAME not in models:
        available = ", ".join(models) if models else "No models found"

        raise RuntimeError(
            f"Required model '{MODEL_NAME}' is not installed.\n"
            f"Available models: {available}\n"
            f"Run: ollama pull {MODEL_NAME}"
        )

    print("[OK] Ollama is reachable.")
    print(f"[OK] Model available: {MODEL_NAME}")


# ============================================================
# LOCAL LLM REQUEST
# ============================================================

def call_local_llm(system_prompt, user_prompt):
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "keep_alive": "10m",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_OUTPUT_TOKENS,
            "num_ctx": 4096
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("[INFO] Sending evidence to the local model...")
    print("[INFO] CPU inference can take several minutes. Please wait.")

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

    except (socket.timeout, TimeoutError) as error:
        raise RuntimeError(
            f"The local model did not finish within "
            f"{REQUEST_TIMEOUT} seconds.\n"
            "Try closing other heavy applications and running "
            "the script again."
        ) from error

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}:\n{details}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach the Ollama API: {error}"
        ) from error

    answer = result.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response. "
            "Try running the model manually with ollama."
        )

    print("[OK] The model generated a response.")
    return answer


# ============================================================
# MITRE RETRIEVAL
# ============================================================

def get_mitre_candidates(mitre_data):
    results = mitre_data.get("results", {})

    behavior = results.get("behavior_focused", {})
    candidates = behavior.get("top_5", [])

    if candidates:
        return candidates

    # Fallback to another available retrieval group.
    for section in results.values():
        if isinstance(section, dict):
            items = section.get("top_5", [])
            if items:
                return items

    return []


# ============================================================
# PROMPTS
# ============================================================

def build_system_prompt():
    return """
You are ZeroShield AI, a defensive SOC analysis assistant.

Analyze only the evidence supplied in the user message.

MANDATORY RULES:
- Never invent IP addresses, hosts, packet captures, timestamps,
  events, or investigation results.
- Distinguish the dataset ground-truth label from the detector prediction.
- Preserve the supplied anomaly score and threshold.
- The detector threshold is unverified for this checkpoint.
- XAI contributions come from replacing one standardized feature at
  a time with zero, representing the standardized training mean.
- These contributions show local perturbation sensitivity, not causality.
- Feature values are standardized or engineered values. Do not describe
  them as raw packet counts or raw network rates.
- MITRE candidates are possible mappings, not confirmed techniques.
- TF-IDF similarity scores are retrieval similarity, not probabilities.
- Do not invent MITRE URLs. Use only supplied URLs.
- Separate known facts from interpretations and recommendations.
- Recommend defensive validation steps, but do not claim they were done.
- You have not inspected live traffic, endpoints, or external systems.

Write a concise Markdown report with these sections:

# ZeroShield AI - LLM-Assisted SOC Report
## 1. Executive Summary
## 2. Sample and Detector Status
## 3. Explainable AI Evidence
## 4. MITRE ATT&CK Candidate Mapping
## 5. Recommended Analyst Validation
## 6. Limitations

Include available technique IDs, names, similarity scores, and source
URLs. Explain that a candidate requires further validation.

Keep the report concise, ideally 400-600 words maximum.
"""


def build_user_prompt(report, candidates):
    evidence = {
        "task": (
            "Generate a concise SOC report grounded in the following "
            "XAI report and MITRE ATT&CK retrieval candidates."
        ),
        "xai_report": report,
        "mitre_candidates": candidates
    }

    return (
        "Use the JSON evidence below. Do not add unsupported facts.\n\n"
        + json.dumps(evidence, ensure_ascii=False)
    )


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def first_available(data, *keys, default="Not available"):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return default


def build_markdown(output):
    return f"""# ZeroShield AI - Local LLM SOC Assistant

- **Generated:** {output['generated_at']}
- **Local model:** `{output['model']}`
- **Execution:** Local Ollama inference
- **API key used:** No
- **Dataset label:** `{output['dataset_label']}`
- **Detector prediction:** `{output['predicted_anomaly']}`
- **MITRE candidates supplied:** {output['mitre_candidate_count']}

---

{output['llm_report_markdown']}

---

## Evidence and Provenance Notes

- This report was generated from the existing XAI report and MITRE retrieval results.
- MITRE mappings are candidates, not confirmation of an attack technique.
- TF-IDF similarity is not an attack probability.
- XAI perturbation contributions are not causal proof.
- The detector threshold verification status is preserved from the input report.
- Verify interpretations against the underlying evidence before using them operationally.
"""


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" ZeroShield AI - Local LLM SOC Assistant")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check the model service first.
    check_ollama()

    # Load the existing evidence.
    print("\n[1/4] Loading existing SOC report...")
    report = load_json(INPUT_REPORT)

    print("[2/4] Loading MITRE retrieval results...")
    mitre_data = load_json(MITRE_FILE)
    candidates = get_mitre_candidates(mitre_data)

    if not candidates:
        raise RuntimeError(
            "No MITRE candidates were found. Run the MITRE retrieval "
            "evaluation script first."
        )

    print(f"[OK] Loaded {len(candidates)} MITRE candidates.")

    # Generate the report.
    print("\n[3/4] Generating report with the local LLM...")

    llm_report = call_local_llm(
        build_system_prompt(),
        build_user_prompt(report, candidates)
    )

    # Preserve input metadata without inventing missing values.
    output = {
        "project": "ZeroShield AI",
        "model": MODEL_NAME,
        "execution": "local_ollama",
        "api_key_used": False,
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset_label": first_available(
            report, "dataset_label", "label"
        ),
        "binary_label": report.get("binary_label"),
        "predicted_anomaly": report.get("predicted_anomaly"),
        "anomaly_score": report.get("anomaly_score"),
        "threshold": report.get("threshold"),
        "threshold_verified_for_checkpoint": report.get(
            "threshold_verified_for_checkpoint", False
        ),
        "mitre_candidate_count": len(candidates),
        "mitre_candidates_supplied": candidates,
        "llm_report_markdown": llm_report
    }

    # Save the results.
    print("[4/4] Saving output files...")

    save_json(OUTPUT_JSON, output)
    OUTPUT_MD.write_text(build_markdown(output), encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUCCESS - SOC report generated!")
    print("=" * 60)
    print(f"\nJSON output: {OUTPUT_JSON}")
    print(f"Markdown output: {OUTPUT_MD}")

    print("\n--- REPORT PREVIEW ---\n")
    print(llm_report)

    print("\n[OK] Open soc_reports/sample_0_llm_soc_report.md in VS Code.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted execution.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(
            f"\n[UNEXPECTED ERROR] {type(error).__name__}: {error}",
            file=sys.stderr
        )
        sys.exit(1)
