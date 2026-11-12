"""Stage 6 live verifier: damaged reports must be caught, honest reports must pass."""
import copy
import os
import unittest
from unittest.mock import patch

from services.llm_service import reason
from services.verification_service import verify_record

TOP = [{"feature": "SYNRatio", "value": 2.4, "contribution": 0.12, "rank": 1},
       {"feature": "PacketDensity", "value": 3.1, "contribution": 0.08, "rank": 2},
       {"feature": "ACKRatio", "value": -1.2, "contribution": 0.03, "rank": 3}]
NAMES = ["SYNRatio", "PacketDensity", "ACKRatio", "RSTRatio", "Rate"]
MITRE = {"techniques": [{"technique_id": "T1498.001", "source": "https://attack.mitre.org/techniques/T1498/001"},
                        {"technique_id": "T1499", "source": "https://attack.mitre.org/techniques/T1499"}],
         "message": "Retrieved ATT&CK candidates are contextual suggestions, not confirmed classifications."}


def analysis(detected=True):
    return {"anomaly_score": 0.2543 if detected else 0.0211, "threshold": 0.0614, "detected": detected,
            "top_features": TOP, "feature_names": [t["feature"] for t in TOP], "all_feature_names": NAMES}


def base_report(detected=True):
    with patch.dict(os.environ, {}, clear=True):
        return reason(analysis(detected), None, MITRE)


def verdict(edit, detected=True, mode=None):
    reasoning = copy.deepcopy(base_report(detected))
    edit(reasoning["report"])
    if mode: reasoning["mode"] = mode
    return verify_record(analysis(detected), MITRE, reasoning)


def append_evidence(text): return lambda r: r["evidence"].append(text)
def set_field(**kw): return lambda r: r.update(kw)


class DamagedReportsAreCaught(unittest.TestCase):
    def assert_caught(self, edit, check_name=None, detected=True):
        result = verdict(edit, detected)
        self.assertNotEqual(result["verdict"], "VERIFIED", result["failed_checks"])
        if check_name: self.assertIn(check_name, result["failed_checks"])

    def test_status_contradicting_detector_when_detected(self):
        self.assert_caught(set_field(incident_summary="The flow did not exceed the threshold and is not anomalous.",
                                     interpretation="Normal traffic."), "threshold_and_status")

    def test_not_anomalous_alone_is_caught(self):
        self.assert_caught(set_field(interpretation="The flow is not anomalous."), "threshold_and_status")

    def test_below_threshold_claim_when_detected(self):
        self.assert_caught(append_evidence("The score is below the threshold."), "threshold_and_status")

    def test_flagged_claim_when_not_detected(self):
        self.assert_caught(set_field(interpretation="The flow was flagged as anomalous by the detector."),
                           "threshold_and_status", detected=False)

    def test_exceeded_claim_when_not_detected(self):
        self.assert_caught(append_evidence("The score exceeded the threshold."), "threshold_and_status", detected=False)

    def test_invented_technique(self):
        self.assert_caught(set_field(attack_context="Consistent with T1059 command execution."), "technique_ids_retrieved")

    def test_fabricated_number(self):
        self.assert_caught(append_evidence("The score was 0.9 for this flow."), "numeric_claims_grounded")

    def test_fake_rounding_of_a_real_number(self):
        self.assert_caught(append_evidence("The threshold was about 0.1."), "numeric_claims_grounded")

    def test_wrong_direction(self):
        self.assert_caught(append_evidence("Replacing SYNRatio with zero increased the anomaly score."),
                           "feature_directions_consistent")

    def test_feature_outside_xai_evidence(self):
        self.assert_caught(append_evidence("Rate was extremely high."), "feature_claims_supported")

    def test_certainty_overclaim(self):
        self.assert_caught(set_field(interpretation="This confirms a Mirai botnet attack."), "certainty_claims_hedged")

    def test_completed_action_claim(self):
        self.assert_caught(set_field(recommended_action="The host was isolated and the flow blocked."),
                           "no_completed_action_claims")

    def test_invented_ip(self):
        self.assert_caught(set_field(interpretation="Traffic came from 192.168.1.45."), "no_invented_specifics")

    def test_invented_mac(self):
        self.assert_caught(set_field(interpretation="Source device aa:bb:cc:dd:ee:ff sent the flow."), "no_invented_specifics")

    def test_invented_date(self):
        self.assert_caught(set_field(interpretation="The flow was captured on 2026-03-14."), "no_invented_specifics")

    def test_invented_url(self):
        self.assert_caught(set_field(attack_context="See https://evil.example.com/report."), "no_invented_specifics")

    def test_limitations_removed(self):
        self.assert_caught(set_field(limitations=""), "uncertainty_present")


class HonestReportsStillPass(unittest.TestCase):
    def test_template_report_verifies_in_both_detector_states(self):
        for detected in (True, False):
            result = verify_record(analysis(detected), MITRE, base_report(detected))
            self.assertEqual(result["verdict"], "VERIFIED", (detected, result["failed_checks"]))

    def test_hedged_and_negated_sentences_are_not_flagged(self):
        for sentence in ("This does not confirm an attack; it requires further validation.",
                         "No response has been carried out and the host was not isolated.",
                         "The retrieved techniques may not be confirmed classifications."):
            self.assertEqual(verdict(append_evidence(sentence))["verdict"], "VERIFIED", sentence)

    def test_legitimate_rounding_and_known_technique_and_source_url_pass(self):
        for sentence in ("The score was about 0.254.", "The threshold was about 0.06.",
                         "Candidate technique T1498.001 was retrieved.",
                         "Source: https://attack.mitre.org/techniques/T1499"):
            self.assertEqual(verdict(append_evidence(sentence))["verdict"], "VERIFIED", sentence)

    def test_not_detected_report_may_say_normal_and_did_not_exceed(self):
        result = verdict(append_evidence("The flow looks normal and did not exceed the threshold."), detected=False)
        self.assertEqual(result["verdict"], "VERIFIED", result["failed_checks"])


class ReasoningModeIsLabelled(unittest.TestCase):
    def test_template_reasoning_carries_a_limitation(self):
        result = verify_record(analysis(), MITRE, base_report())
        self.assertEqual(result["reasoning_mode"], "Demo Fallback Reasoning")
        self.assertIn("internal consistency", result["limitation"])

    def test_llm_reasoning_has_no_template_limitation(self):
        result = verdict(lambda r: None, mode="OpenAI LLM")
        self.assertEqual(result["reasoning_mode"], "OpenAI LLM")
        self.assertNotIn("limitation", result)


if __name__ == "__main__": unittest.main()
