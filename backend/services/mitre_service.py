import hashlib
import json
import logging
from pathlib import Path

# ---- Behaviour ontology -------------------------------------------------------------------------
# ATT&CK descriptions are written in plain language ("high volume of packets", "SYN flood", "scanning").
# The old query was feature names and numbers ("syn_count value -1.0099 ..."), which share no words with
# those descriptions, so every alert retrieved the same irrelevant techniques. This map translates a feature
# that is clearly outside the benign range into the behaviour it signals. It is written from networking
# knowledge, not fitted to labels; the query never contains a dataset label.
# Each entry: (neutral name, phrase when value is high, phrase when value is low).
BEHAVIOR_ONTOLOGY = {
    "flow_duration": ("connection duration", "long lasting persistent connection", "very short lived connections"),
    "Header_Length": ("packet header length", "large packet headers fragmented or tunneled packets", "minimal packet headers"),
    "Protocol Type": ("network protocol", "unusual non-standard network protocol", "unusual non-standard network protocol"),
    "Duration": ("packet time to live", "unusual packet time to live", "unusual packet time to live"),
    "Rate": ("packet rate", "high rate of packets high volume of traffic flood overwhelming a network service denial of service", "low packet rate slow traffic"),
    "Drate": ("inbound packet rate", "high inbound packet rate", "no inbound responses one-way traffic"),
    "IAT": ("gaps between packets", "long gaps between packets periodic beaconing", "very short gaps between packets rapid bursts of packets high volume flood"),
    "Number": ("packet count", "many packets", "few packets"),
    "AVG": ("average packet size", "large packets", "small packets minimal payload"),
    "Std": ("packet size variation", "highly variable packet sizes", "uniform identical packet sizes automated traffic"),
    "Radius": ("packet size spread", "large variation in packet size", "uniform packet sizes automated traffic"),
    "Weight": ("traffic volume", "large traffic volume", "small traffic volume"),
    "PacketDensity": ("packet density", "very high packet density in a very short time high volume flood denial of service", "low packet density"),
    "syn_flag_number": ("TCP SYN flag", "many TCP SYN packets connection requests SYN flood scanning network services", "no TCP SYN packets"),
    "syn_count": ("SYN packet count", "many TCP SYN packets connection requests SYN flood scanning network services", "no TCP SYN packets"),
    "SYNRatio": ("share of SYN packets", "mostly TCP SYN packets connection requests SYN flood scanning network services", "few TCP SYN packets"),
    "SYN_ACK_Ratio": ("SYN to ACK ratio", "many SYN packets without ACK replies SYN flood", "balanced SYN and ACK"),
    "ack_flag_number": ("TCP ACK flag", "many TCP ACK packets", "no TCP acknowledgement one-way traffic"),
    "ack_count": ("ACK packet count", "many TCP ACK packets", "no TCP acknowledgement one-way traffic"),
    "ACKRatio": ("share of ACK packets", "mostly TCP ACK packets", "no TCP acknowledgement one-way traffic"),
    "rst_flag_number": ("TCP RST flag", "many TCP RST reset packets connection refused probing closed ports network scanning", "no TCP reset packets"),
    "rst_count": ("RST packet count", "many TCP RST reset packets connection refused probing closed ports network scanning", "no TCP reset packets"),
    "RSTRatio": ("share of RST packets", "mostly TCP RST reset packets probing closed ports network scanning", "few TCP reset packets"),
    "fin_flag_number": ("TCP FIN flag", "many TCP FIN packets", "no TCP FIN packets"),
    "fin_count": ("FIN packet count", "many TCP FIN packets", "no TCP FIN packets"),
    "FINRatio": ("share of FIN packets", "mostly TCP FIN packets", "few TCP FIN packets"),
    "psh_flag_number": ("TCP PSH flag", "many TCP PSH push packets application data flood", "no TCP push packets"),
    "urg_count": ("TCP urgent flag", "TCP urgent flag set unusual packets", "no TCP urgent flag"),
    "ece_flag_number": ("TCP ECE flag", "TCP congestion flag set", "no TCP congestion flag"),
    "cwr_flag_number": ("TCP CWR flag", "TCP congestion flag set", "no TCP congestion flag"),
    "HTTP": ("HTTP web traffic", "HTTP web requests to a public-facing web application web server", "no web traffic"),
    "HTTPS": ("HTTPS web traffic", "encrypted HTTPS web requests to a web application", "no encrypted web traffic"),
    "DNS": ("DNS traffic", "DNS queries and responses name resolution", "no DNS traffic"),
    "Telnet": ("Telnet traffic", "Telnet remote login credentials brute force", "no Telnet traffic"),
    "SSH": ("SSH traffic", "SSH remote login credentials brute force", "no SSH traffic"),
    "SMTP": ("SMTP traffic", "SMTP mail protocol", "no SMTP traffic"),
    "IRC": ("IRC traffic", "IRC chat protocol command and control botnet", "no IRC traffic"),
    "DHCP": ("DHCP traffic", "DHCP address assignment", "no DHCP traffic"),
    "ARP": ("ARP traffic", "many ARP requests and replies ARP spoofing adversary in the middle network discovery", "no ARP traffic"),
    "ICMP": ("ICMP traffic", "many ICMP echo messages ping sweep network discovery flood", "no ICMP traffic"),
    "TCP": ("TCP traffic", "TCP connections", "not TCP traffic"),
    "UDP": ("UDP traffic", "UDP datagrams UDP flood", "not UDP traffic"),
    "IPv": ("IP traffic", "IP traffic", "non IP traffic"),
}
# The absence of a protocol is not attack behaviour, so protocol indicators only contribute when they are present.
# (Dropping their "no X traffic" phrases left the relevant results unchanged on the tuning half and cut wrong answers.)
for _name in ("HTTP", "HTTPS", "DNS", "Telnet", "SSH", "SMTP", "IRC", "DHCP", "ARP", "ICMP", "IPv", "TCP", "UDP"):
    _neutral, _high, _low = BEHAVIOR_ONTOLOGY[_name]
    BEHAVIOR_ONTOLOGY[_name] = (_neutral, _high, None)
HIGH_Z, LOW_Z = 1.0, -1.0              # "clearly outside the benign range", in scaled units
MIN_SIMILARITY = 0.15                  # chosen on a tuning half of the labelled evidence, reported on the other half
# Flow data can only support techniques that are visible on the network. Persistence, privilege-escalation,
# defense-evasion and execution techniques (registry keys, plist files, ...) cannot be inferred from flow statistics.
NETWORK_TACTICS = frozenset({"reconnaissance", "initial-access", "discovery", "credential-access", "collection",
                             "command-and-control", "exfiltration", "impact", "lateral-movement"})
RETRIEVAL_METHOD = "behaviour-phrase TF-IDF cosine similarity, restricted to network-observable ATT&CK tactics"
RELIABILITY_NOTE = ("Flow statistics describe volume, timing, flags and protocols. They can suggest network-level techniques "
                    "such as floods, but they cannot identify application-layer techniques (injection, XSS and similar); "
                    "treat candidates for such flows as low confidence. The five candidates score close together, so their order is not a ranking.")


def build_behavior_query(top_features, top_n: int = 8):
    """Plain-language behaviours for the features that push the score up and sit clearly outside the benign range."""
    phrases, used = [], []
    for item in list(top_features)[:top_n]:
        entry = BEHAVIOR_ONTOLOGY.get(item.get("feature"))
        if not entry or float(item.get("contribution", 0)) <= 0:
            continue
        value = float(item["value"])
        phrase = entry[1] if value >= HIGH_Z else entry[2] if value <= LOW_Z else None
        if phrase and phrase not in phrases:
            phrases.append(phrase)
            used.append({"feature": item["feature"], "value": value, "contribution": float(item["contribution"]), "behaviour": phrase})
    return ("network traffic with " + ", ".join(phrases) if phrases else ""), used


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
                self._verify_index_hash(index_path)
                with index_path.open("rb") as handle:
                    self.index = pickle.load(handle)
        except Exception as exc:
            self.index_error = str(exc)

    @staticmethod
    def _verify_index_hash(path: Path):
        """Pickle can execute code, so refuse to load an index whose SHA-256 differs from the recorded one.
        The sidecar is written by LLM/build_mitre_rag.py. Without it the index loads with a warning."""
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            logging.warning("%s not found; loading the TF-IDF index without an integrity check.", sidecar.name)
            return
        expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"TF-IDF index hash mismatch (expected {expected[:12]}, got {actual[:12]}); refusing to unpickle it.")

    @property
    def available(self):
        return bool(self.records)

    def insufficient(self):
        return {"status": "insufficient_evidence", "message": "Insufficient evidence for a confident ATT&CK mapping.", "techniques": []}

    def retrieve_behavior(self, analysis):
        if not self.index:
            return {**self.insufficient(), "status": "retrieval_unavailable", "message": "Behavior retrieval unavailable; no techniques fabricated.", "error": self.index_error}
        query, used = build_behavior_query(analysis.get("top_features", []))
        base = {"query": query, "evidence_features": used, "method": RETRIEVAL_METHOD,
                "similarity_floor": MIN_SIMILARITY, "reliability_note": RELIABILITY_NOTE}
        if not query:
            return {**self.insufficient(), **base, "message": "No feature was clearly outside the benign range, so no behaviour could be described and no ATT&CK mapping was attempted."}
        store = self.index
        try:
            scores = (store["vectorizer"].transform([query]) @ store["matrix"].T).toarray().ravel()
            results = []
            for index in scores.argsort()[::-1]:
                if scores[index] < MIN_SIMILARITY: break
                doc = store["documents"][int(index)]
                if not (set(doc.get("tactics", [])) & NETWORK_TACTICS): continue
                results.append({"technique_id": doc["id"], "technique_name": doc["name"],
                                "description": doc["text"].split("Description: ", 1)[-1][:1200],
                                "source": doc.get("url"), "similarity": float(scores[index]),
                                "tactics": doc.get("tactics", []), "platforms": doc.get("platforms", [])})
                if len(results) == 5: break
        except Exception as exc:
            return {**self.insufficient(), **base, "status": "retrieval_unavailable", "message": "Behavior retrieval failed; no techniques fabricated.", "error": str(exc)}
        return {**base, "status": "retrieved" if results else "empty_retrieval", "techniques": results,
                "message": "Retrieved ATT&CK candidates are contextual suggestions, not confirmed classifications." if results else "No ATT&CK technique matched the observed behaviour strongly enough."}

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
