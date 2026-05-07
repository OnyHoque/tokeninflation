# Code Review Package

This folder contains documented and anonymized versions of the submitted experiment scripts.

## Files

- `data_processing.py` — filters a JSON dataset to examples where `label`, `rule_score`, and `learning_score` agree.
- `library_v1.py` — shared verifier utilities, scoring functions, RNN/DeepSet models, and the `CoInVerifier` wrapper.
- `SingleRun_L.py` — thin wrapper for the learned RNN verifier.
- `SingleRun_R.py` — thin wrapper for the rule-based verifier.
- `seq2seq_gen_qwen.py` — LoRA fine-tuning script for answer-to-synthetic-block generation.
- `seq2seq_gen_T5.py` — T5 generation and verifier-scoring script.


## Important dependency note

The package still expects the original local model/checkpoint directories and the project-specific `heads.py` implementation to be available at runtime. Please use the official CoIn library to get the model weights and code: https://github.com/CASE-Lab-UMD/LLM-Auditing-CoIn