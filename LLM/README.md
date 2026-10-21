# ZeroShield AI - Teammate Handoff

## Project pipeline
Deep SVDD anomaly detection
-> XAI feature explanations
-> MITRE ATT&CK retrieval
-> Local Ollama LLM SOC report
-> Post-LLM verification (next implementation stage)

## Requirements
- Python 3.11 or 3.12 recommended
- PyTorch
- NumPy
- pandas
- scikit-learn
- Ollama, if running the local LLM assistant
- Ollama model: llama3.2:3b

## Setup
Create a Python virtual environment and install compatible packages.
Check each script's imports before running it.

Install Ollama separately and run:
ollama pull llama3.2:3b

Start Ollama and check:
http://localhost:11434

## Important limitations
- The existing threshold 0.06141921 is provisional, not validated.
- MITRE TF-IDF similarity scores are not probabilities.
- XAI feature sensitivities are not proof of causation.
- Existing reports are example outputs, not proof of detector accuracy.
- Large datasets are intentionally excluded from this handoff.
- Scripts that read files under Dataset/ need the required dataset files.
- Run scripts from the project root so relative paths resolve correctly.

## Next task
Implement post-LLM verification:
1. Parse the LLM report into a defined schema.
2. Check claims against detector scores, XAI evidence and retrieved MITRE data.
3. Flag unsupported claims and invalid technique IDs.
4. Produce a verified report with pass/fail results and reasons.
5. Keep the original LLM output for comparison and auditing.
