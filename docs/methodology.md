# ZeroShield AI methodology and reproducibility

## Feature preparation and detection

The existing research scripts use prepared feature CSVs and a benign-only Deep SVDD implementation. The production demo adapter loads `model_training/outputs/Deep_SVDD_Tuned.pt` on CPU and reads feature order, network dimensions, center, and threshold from the checkpoint. The active representation has 43 features. The score is the sum of squared distances between the network embedding and stored center; the frozen decision is anomalous when score is at least the stored threshold (`0.06141921002417763`).

The repository does not contain a fitted scaler object or an API transform from raw flow fields to the exact prepared representation. The API therefore accepts only values already prepared in that representation. The small demo cache contains prepared-space rows; it is for a walkthrough, not a performance estimate.

## Explanation and context

For each input, the backend replaces one feature value with zero in prepared space and recomputes the score. The difference between original and perturbed score is reported as feature sensitivity. It is not a causal attribution. MITRE retrieval builds text from observed top features, their values and contribution directions, and score/threshold; TF-IDF returns ranked candidates. The category mapping file is a separate human-curated contextual source and must not be treated as model prediction.

## Reasoning, verification, and response

When `OPENAI_API_KEY` exists, the backend sends evidence to OpenAI and requests structured JSON. Otherwise it returns a clearly labeled deterministic fallback. The verifier checks that claims remain consistent with the supplied score, threshold, feature evidence, and retrieved technique IDs. `VERIFIED` means evidence consistency under these implemented checks, not that the underlying alert is a true attack.

LinUCB uses pre-action context and one of four simulated actions. Ground truth is used only in feedback to calculate the documented reward and update policy. No network or host operation is performed. Experimental rewards and plots describe the designed offline simulation only.

## Reproduction and limits

Install the dependencies and run commands in the root README. The active model checkpoint and MITRE index are small runtime artifacts; the index can be rebuilt from the checked-in technique JSON using the builder from the `LLM` directory. The large datasets are excluded and are unnecessary for application startup. Reproducing training or evaluation needs separately authorized dataset access, the exact original preparation/scaler process, and substantial storage; those provenance details are not fully represented in the repository. Do not interpret existing reports as independent validation of the current detector.
