
import json
import time
from pathlib import Path

import requests


# Official MITRE-maintained ATT&CK STIX data
BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/"
    "attack-stix-data/master/enterprise-attack/enterprise-attack.json"
)

CACHE_DIR = Path("data")
CACHE_FILE = CACHE_DIR / "mitre_enterprise_techniques.json"


def download_bundle():
    """Download the official Enterprise ATT&CK STIX bundle."""
    print("Downloading official MITRE ATT&CK Enterprise STIX data...")

    response = requests.get(
        BUNDLE_URL,
        timeout=(15, 180),
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()

    bundle = response.json()

    if bundle.get("type") != "bundle" or not isinstance(
        bundle.get("objects"), list
    ):
        raise ValueError("Downloaded data is not a valid STIX bundle.")

    print(f"Download successful: {len(response.content):,} bytes")
    return bundle


def extract_techniques(bundle):
    """Extract non-revoked, non-deprecated ATT&CK techniques."""
    techniques = {}

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue

        if obj.get("revoked", False):
            continue

        if obj.get("x_mitre_deprecated", False):
            continue

        external_id = None
        attack_url = None

        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                external_id = ref.get("external_id")
                attack_url = ref.get("url")
                break

        # Ignore objects without a standard ATT&CK technique ID.
        if not external_id or not external_id.startswith("T"):
            continue

        technique = {
            "id": external_id,
            "name": obj.get("name", ""),
            "description": obj.get("description", ""),
            "url": attack_url or "",
            "stix_id": obj.get("id", ""),
            "modified": obj.get("modified", ""),
            "platforms": obj.get("x_mitre_platforms", []),
            "tactics": sorted({
                phase.get("phase_name", "")
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("kill_chain_name") == "mitre-attack"
                and phase.get("phase_name")
            }),
            "is_subtechnique": obj.get(
                "x_mitre_is_subtechnique", False
            ),
        }

        techniques[external_id] = technique

    return techniques


def update_cache():
    """Download, extract and save the local technique cache."""
    bundle = download_bundle()
    techniques = extract_techniques(bundle)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = CACHE_FILE.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            techniques,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(CACHE_FILE)

    print(f"Extracted techniques: {len(techniques):,}")
    print(f"Cache saved to: {CACHE_FILE.resolve()}")

    return techniques


def load_techniques(refresh=False):
    """Load cached techniques, optionally refreshing from MITRE's repository."""
    if refresh or not CACHE_FILE.exists():
        return update_cache()

    with CACHE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def lookup_technique(technique_id, refresh=False):
    """Look up an ATT&CK technique, for example T1498.001."""
    techniques = load_techniques(refresh=refresh)
    return techniques.get(technique_id.upper())


def main():
    techniques = update_cache()

    for technique_id in ("T1498.001", "T1190", "T1059"):
        technique = techniques.get(technique_id)

        print(f"\n--- {technique_id} ---")
        if technique:
            print("Name:", technique["name"])
            print("URL:", technique["url"])
            print("Tactics:", ", ".join(technique["tactics"]))
            print("Platforms:", ", ".join(technique["platforms"]))
        else:
            print("Technique not found in the current cache.")


if __name__ == "__main__":
    main()
