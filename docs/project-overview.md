# ZeroShield AI project overview

ZeroShield AI is a research prototype for behavior-based network-flow anomaly detection in a SOC-style workflow. Its runtime joins an existing benign-only Deep SVDD checkpoint with local feature perturbation explanations, MITRE ATT&CK retrieval, optional evidence-bound LLM reasoning, deterministic verification checks, and an offline contextual response policy.

## Runtime flow

```text
prepared 43-feature flow
  -> Deep SVDD score and stored threshold
  -> zero-baseline perturbation evidence
  -> behavior-derived TF-IDF ATT&CK candidates
  -> OpenAI reasoning when configured, otherwise labeled fallback
  -> structured evidence verification
  -> simulated LinUCB response and feedback ledger
  -> FastAPI + React dashboard
```

The model score is squared distance from the learned embedding center. The checkpoint stores threshold `0.06141921002417763`; scores greater than or equal to it are flagged. XAI measures score sensitivity when each feature is replaced by zero in prepared space. These outputs support analyst review and do not establish causation or a confirmed attack.

## Project areas

- `backend/`: FastAPI orchestration, detector adapter, retrieval, LLM, verification, response policy, tests, and small demo cache.
- `frontend/`: React/Vite user interface and package lockfile.
- `model_training/`: original scripts, active checkpoint, feature definition, and selected research outputs.
- `LLM/`: retrieval builder and technique corpus, active retrieval index, related research utilities.
- `ZeroShield_Verification_Handoff/`: standalone report verifier and fixtures.
- `Documentation/`: project reports and presentations.

## Run and limitations

Follow the exact Windows commands in the root [README](../README.md). The application requires the active checkpoint, feature list, mapping, technique source/index, and demo fixture described there. The large research CSVs are not required at runtime. Inputs must already be in prepared/scaled space because a fitted scaler and raw preprocessing transform were not found. OpenAI is optional; no real response action is executed. The app is a prototype and does not provide production monitoring, real containment, or validated operational performance claims.
