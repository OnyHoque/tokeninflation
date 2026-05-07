"""Fine-tune a causal LM to generate synthetic reasoning blocks from answers.

The script trains a LoRA adapter on answer/block pairs and periodically evaluates
sampled generations with the same verifier used in the experiments.

Expected training data format
-----------------------------
The input JSON file should contain a list of dictionaries with these keys:

``answer``
    The final answer or answer block.
``block``
    The target reasoning block associated with that answer.

Example
-------
python seq2seq_gen_qwen.py \
    --data-path data/train_5K_chunked_256.json \
    --output-dir outputs/qwen_synthetic_block_lora
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from library_v1 import CoInVerifier, DEFAULT_MODEL_A_DIR, DEFAULT_MODEL_B_DIR


KEY_ANSWER = "answer"
KEY_BLOCK = "block"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSON training data."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data[:max_samples] if max_samples else data


def split_data(data: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shuffle and split data into train/validation partitions."""
    shuffled = data[:]
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]


def build_prompt(answer_text: str) -> str:
    """Construct the supervised fine-tuning prompt."""
    return (
        "Generate a synthetic reasoning block that is semantically aligned with the answer block.\n\n"
        f"Answer block:\n{answer_text.strip()}\n\n"
        "Synthetic reasoning block:\n"
    )


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device where ``model`` stores its parameters."""
    return next(model.parameters()).device


def normalize_block_length(verifier: CoInVerifier, text: str, block_tokens: int = 256) -> str:
    """Truncate or pad generated text to exactly ``block_tokens`` verifier tokens."""
    tokenizer = verifier.infer_b.embedding_model.tokenizer
    valid_ids = list(verifier.infer_b.tokenid2emb.keys())
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > block_tokens:
        token_ids = token_ids[:block_tokens]
    else:
        token_ids += [random.choice(valid_ids) for _ in range(block_tokens - len(token_ids))]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.inference_mode()
def generate_block(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    answer_text: str,
    max_total_tokens: int,
    max_block_tokens: int,
) -> str:
    """Generate one synthetic reasoning block from an answer."""
    model.eval()
    prompt = build_prompt(answer_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_total_tokens - max_block_tokens - 1,
    )
    inputs = {key: value.to(get_model_device(model)) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_block_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def score_block(verifier: CoInVerifier, block_text: str, answer_text: str) -> Tuple[float, float]:
    """Return block-to-answer and token-to-block scores."""
    score = verifier.block_score(block_text, answer_text, verbose=False)[0]
    return float(score["b2a"]), float(score["t2b"])


class AnswerToBlockCausalDataset(Dataset):
    """Causal-LM dataset for answer-to-block supervised fine-tuning."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_block_tokens: int,
        max_total_tokens: int,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_block_tokens = max_block_tokens
        self.max_total_tokens = max_total_tokens
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        example = self.data[index]
        answer = example[KEY_ANSWER].strip()
        block = example[KEY_BLOCK].strip()
        prompt_ids = self.tokenizer(
            build_prompt(answer),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_total_tokens - self.max_block_tokens - 1,
        )["input_ids"]
        remaining = self.max_total_tokens - len(prompt_ids) - 1
        target_ids = self.tokenizer(
            block,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, min(self.max_block_tokens, remaining)),
        )["input_ids"]

        input_ids = prompt_ids + target_ids + [self.eos_id]
        labels = [-100] * len(prompt_ids) + target_ids + [self.eos_id]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_total_tokens - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class VerifierEvalCallback(TrainerCallback):
    """Evaluate sampled generations with verifier scores during training."""

    def __init__(
        self,
        verifier: CoInVerifier,
        tokenizer: AutoTokenizer,
        val_data: List[Dict[str, Any]],
        sample_n: int,
        max_total_tokens: int,
        max_block_tokens: int,
    ) -> None:
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.val_data = val_data
        self.sample_n = sample_n
        self.max_total_tokens = max_total_tokens
        self.max_block_tokens = max_block_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        sample = random.sample(self.val_data, min(self.sample_n, len(self.val_data)))
        passed = 0
        b2a_total = 0.0
        t2b_total = 0.0

        for example in sample:
            raw_block = generate_block(
                model,
                self.tokenizer,
                example[KEY_ANSWER],
                self.max_total_tokens,
                self.max_block_tokens,
            )
            block = normalize_block_length(self.verifier, raw_block, block_tokens=self.max_block_tokens)
            b2a_score, t2b_score = score_block(self.verifier, block, example[KEY_ANSWER])
            b2a_total += b2a_score
            t2b_total += t2b_score
            passed += int(b2a_score >= 0.5 and t2b_score >= 0.5)

        n = max(1, len(sample))
        print(f"\n[VerifierEval] pass={passed/n:.3f} b2a={b2a_total/n:.3f} t2b={t2b_total/n:.3f}\n")
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data-path", default="data/train_5K_chunked_256.json")
    parser.add_argument("--output-dir", default="outputs/qwen_synthetic_block_lora")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-block-tokens", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=640)
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-sample-n", type=int, default=24)
    parser.add_argument("--model-a-dir", default=DEFAULT_MODEL_A_DIR)
    parser.add_argument("--model-b-dir", default=DEFAULT_MODEL_B_DIR)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading even when CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = load_data(args.data_path, args.max_samples)
    train_data, val_data = split_data(data, args.val_ratio, args.seed)

    verifier = CoInVerifier(
        model_a_dir=args.model_a_dir,
        model_b_dir=args.model_b_dir,
        verifier_model_type="RNN",
        verifier_hidden_dim=256,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_kwargs: Dict[str, Any] = {}

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if bf16_ok else torch.float16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = AnswerToBlockCausalDataset(train_data, tokenizer, args.max_block_tokens, args.max_total_tokens)
    val_ds = AnswerToBlockCausalDataset(val_data, tokenizer, args.max_block_tokens, args.max_total_tokens)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.log_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16_ok,
        fp16=torch.cuda.is_available() and not bf16_ok,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=[
            VerifierEvalCallback(
                verifier,
                tokenizer,
                val_data,
                args.eval_sample_n,
                args.max_total_tokens,
                args.max_block_tokens,
            )
        ],
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n=== Sample generations scored by verifier ===")
    for example in random.sample(val_data, min(5, len(val_data))):
        raw_block = generate_block(model, tokenizer, example[KEY_ANSWER], args.max_total_tokens, args.max_block_tokens)
        block = normalize_block_length(verifier, raw_block, block_tokens=args.max_block_tokens)
        b2a_score, t2b_score = score_block(verifier, block, example[KEY_ANSWER])
        print("-" * 80)
        print("ANSWER:", example[KEY_ANSWER][:100].replace("\n", " "), "...")
        print("BLOCK :", block[:100].replace("\n", " "), "...")
        print("B2A:", round(b2a_score, 4), "T2B:", round(t2b_score, 4))
        print("PASS:", b2a_score >= 0.5 and t2b_score >= 0.5)


if __name__ == "__main__":
    main()
