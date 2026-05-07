"""Generate and score synthetic reasoning blocks with a trained T5 model.

The script loads a benign evaluation set, appends generated 256-token blocks to
selected samples, and evaluates the modified samples with both the rule-based and
learned verifiers.

Example
-------
python seq2seq_gen_T5.py \
    --data-path data/benign_label1.json \
    --model-path T5_trained_9K_v2 \
    --num-samples 10 \
    --blocks-per-sample 10
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from SingleRun_L import SingleRunClassifier as LearnedVerifierRunner
from SingleRun_R import SingleRunClassifier as RuleVerifierRunner
from library_v1 import DEFAULT_MODEL_A_DIR, DEFAULT_MODEL_B_DIR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a JSON list of evaluation samples."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


@torch.inference_mode()
def generate_block_t5(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    answer_text: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate one synthetic block conditioned on an answer."""
    inputs = tokenizer(
        "synthetic_block: " + answer_text.strip(),
        return_tensors="pt",
        truncation=True,
    ).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def normalize_block_length(infer_b, text: str, block_tokens: int = 256) -> str:
    """Truncate or pad text to exactly ``block_tokens`` verifier tokens."""
    tokenizer = infer_b.embedding_model.tokenizer
    valid_ids = list(infer_b.tokenid2emb.keys())
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > block_tokens:
        token_ids = token_ids[:block_tokens]
    else:
        token_ids += [random.choice(valid_ids) for _ in range(block_tokens - len(token_ids))]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def run_generation_experiment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate blocks, evaluate them, and return structured results."""
    data = load_data(args.data_path)[: args.num_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(DEVICE)

    rule_runner = RuleVerifierRunner(args.model_a_dir, args.model_b_dir, token_ratio=args.token_ratio)
    learned_runner = LearnedVerifierRunner(args.model_a_dir, args.model_b_dir, token_ratio=args.token_ratio)
    infer_b = rule_runner.infer_b

    results: List[Dict[str, Any]] = []
    for sample_index, original_item in enumerate(data):
        item = copy.deepcopy(original_item)
        answer_text = item["A"]

        for block_index in range(args.blocks_per_sample):
            raw_block = generate_block_t5(model, tokenizer, answer_text, max_new_tokens=args.max_new_tokens)
            block = normalize_block_length(infer_b, raw_block, block_tokens=args.block_tokens)
            item["R"] = item["R"] + [block]

            modified_sample = [item]
            rule_score = rule_runner.predict(modified_sample, v_thresh=args.rule_threshold, p_ratio=args.probing_ratio)[0]
            learned_score = learned_runner.predict(modified_sample, v_thresh=args.learned_threshold, p_ratio=args.probing_ratio)[0]

            record = {
                "sample_index": sample_index,
                "block_index": block_index,
                "rule_verifier_pass": int(rule_score),
                "learned_verifier_pass": int(learned_score),
                "generated_block": block,
            }
            results.append(record)
            print(
                f"sample={sample_index} block={block_index} "
                f"rule={rule_score} learned={learned_score}"
            )
            print(block)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/benign_label1.json")
    parser.add_argument("--model-path", default="T5_trained_9K_v2")
    parser.add_argument("--model-a-dir", default=DEFAULT_MODEL_A_DIR)
    parser.add_argument("--model-b-dir", default=DEFAULT_MODEL_B_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--blocks-per-sample", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-tokens", type=int, default=256)
    parser.add_argument("--token-ratio", type=float, default=0.5)
    parser.add_argument("--probing-ratio", type=float, default=0.75)
    parser.add_argument("--rule-threshold", type=float, default=0.5)
    parser.add_argument("--learned-threshold", type=float, default=0.6)
    parser.add_argument("--output", default=None, help="Optional path to save JSON results.")
    return parser.parse_args()


def main() -> None:
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args = parse_args()
    results = run_generation_experiment(args)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
