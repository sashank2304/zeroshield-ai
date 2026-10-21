
import json
from pathlib import Path

CACHE_FILE = Path("data/mitre_enterprise_techniques.json")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "mitre_technique_lookup_report.json"

# Existing project mapping examples discussed previously.
# These are lookup checks, NOT newly asserted attack mappings.
TECHNIQUE_IDS_TO_CHECK = [
    "T1498.001",
    "T1190",
    "T1059",
    "T1105",
    "T1185",
    "T1189",
    "T1505.003",
]


def main():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}. Run python mitre_api.py first."
        )

    with CACHE_FILE.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    report = {
        "source": (
            "Official MITRE ATT&CK Enterprise STIX bundle hosted "
            "in the mitre-attack/attack-stix-data repository"
        ),
        "cache_file": str(CACHE_FILE),
        "techniques_checked": [],
        "not_found": [],
    }

    for technique_id in TECHNIQUE_IDS_TO_CHECK:
        technique = cache.get(technique_id)

        if technique is None:
            report["not_found"].append(technique_id)
            continue

        report["techniques_checked"].append({
            "id": technique_id,
            "name": technique.get("name", ""),
            "description": technique.get("description", ""),
            "url": technique.get("url", ""),
            "tactics": technique.get("tactics", []),
            "platforms": technique.get("platforms", []),
            "modified": technique.get("modified", ""),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Techniques found: {len(report['techniques_checked'])}")
    print(f"Technique IDs not found: {report['not_found']}")
    print(f"Report saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

