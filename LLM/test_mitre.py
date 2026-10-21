
import json
from pathlib import Path

cache_path = Path("data/mitre_enterprise_techniques.json")

if not cache_path.exists():
    raise FileNotFoundError(
        "MITRE cache missing. Run python mitre_api.py first."
    )

with cache_path.open("r", encoding="utf-8") as f:
    techniques = json.load(f)

test_ids = ["T1498.001", "T1190", "T1059"]

print("=" * 65)
print("ZEROSHIELD AI - MITRE ATT&CK LOOKUP TEST")
print("=" * 65)

for technique_id in test_ids:
    item = techniques.get(technique_id)

    print(f"\nTechnique ID: {technique_id}")

    if not item:
        print("RESULT: NOT FOUND")
        continue

    print("RESULT: FOUND")
    print("Name:", item.get("name", ""))
    print("Tactics:", ", ".join(item.get("tactics", [])))
    print("Platforms:", ", ".join(item.get("platforms", [])))
    print("Official URL:", item.get("url", ""))
    print("Description:")
    print(item.get("description", "No description available."))

print("\n" + "=" * 65)
print("Lookup test complete.")
print("Note: finding a technique does not confirm an attack mapping.")
