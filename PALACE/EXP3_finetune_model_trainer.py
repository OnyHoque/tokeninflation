import os
import gc
import json
import inspect
import shutil
import random
from typing import Dict, Any, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PALACE_GPU_ID", "3")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL_DIR = "models/PALACE_GRPO_Qwen2.5-1.5B-General"

DATA_PATHS = [
    "dataset/general_15K_adv.json",
]

SAMPLES_PER_FILE = 15000
OUTPUT_ADAPTER_DIR = "finetuned_adapters/palace_general_main"
OUTPUT_MERGED_DIR = "finetuned_models/palace_general_main15"

MAX_LENGTH = 1024
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42


def load_json_samples(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def sample_rows(samples: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    valid = [s for s in samples if "prompt" in s and "ground_truth" in s]
    if len(valid) <= k:
        return valid
    return rng.sample(valid, k)

def select_first_rows(samples: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    valid = [s for s in samples if "prompt" in s and "ground_truth" in s]
    return valid[:k]


def build_completion(sample: Dict[str, Any]) -> str:
    return json.dumps({"count": int(sample["ground_truth"])}, ensure_ascii=False)


def prepare_rows(samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for s in samples:
        rows.append({
            "prompt": str(s["prompt"]),
            "completion": build_completion(s),
        })
    return rows


def tokenize_example(example, tokenizer, max_length: int):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 64,
    )["input_ids"]

    completion_text = example["completion"] + (tokenizer.eos_token or "")
    completion_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
        truncation=True,
        max_length=64,
    )["input_ids"]

    input_ids = prompt_ids + completion_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must have pad_token_id or eos_token_id")
            pad_id = self.tokenizer.eos_token_id

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def filter_kwargs(fn, kwargs):
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def safe_rmtree(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def model_from_pretrained(path: str, **kwargs):
    kwargs = filter_kwargs(AutoModelForCausalLM.from_pretrained, kwargs)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def tokenizer_from_pretrained(path: str, **kwargs):
    kwargs = filter_kwargs(AutoTokenizer.from_pretrained, kwargs)
    return AutoTokenizer.from_pretrained(path, **kwargs)


def peft_from_pretrained(base_model, adapter_dir: str, **kwargs):
    kwargs = filter_kwargs(PeftModel.from_pretrained, kwargs)
    return PeftModel.from_pretrained(base_model, adapter_dir, **kwargs)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_training_model(base_model_dir: str):
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    if use_cuda:
        torch.cuda.set_device(0)
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = model_from_pretrained(
            base_model_dir,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": 0},
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        compute_dtype = torch.float32
        model = model_from_pretrained(
            base_model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, compute_dtype


def build_training_args(output_dir: str, compute_dtype):
    train_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": SEED,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "no_cuda": not torch.cuda.is_available(),
    }

    if compute_dtype == torch.bfloat16:
        train_args_kwargs["bf16"] = True
    elif compute_dtype == torch.float16:
        train_args_kwargs["fp16"] = True

    if "optim" in inspect.signature(TrainingArguments.__init__).parameters:
        train_args_kwargs["optim"] = "adamw_torch"
    if "report_to" in inspect.signature(TrainingArguments.__init__).parameters:
        train_args_kwargs["report_to"] = "none"

    return TrainingArguments(**filter_kwargs(TrainingArguments.__init__, train_args_kwargs))


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if not os.path.isdir(BASE_MODEL_DIR):
        raise FileNotFoundError(f"Base model dir not found: {BASE_MODEL_DIR}")

    safe_rmtree(OUTPUT_ADAPTER_DIR)
    safe_rmtree(OUTPUT_MERGED_DIR)
    os.makedirs(OUTPUT_ADAPTER_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MERGED_DIR, exist_ok=True)

    combined_samples = []

    for path in DATA_PATHS:
        samples = load_json_samples(path)
        selected = select_first_rows(samples, SAMPLES_PER_FILE)
        combined_samples.extend(selected)
        print(f"Loaded {len(samples)} from {path}")
        print(f"Selected first {len(selected)} from {path}")

    rows = prepare_rows(combined_samples)
    print(f"Total training rows: {len(rows)}")

    dataset = Dataset.from_list(rows)

    tokenizer = tokenizer_from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = dataset.map(
        lambda x: tokenize_example(x, tokenizer, MAX_LENGTH),
        remove_columns=dataset.column_names,
        desc="Tokenizing general",
    )

    model, compute_dtype = load_training_model(BASE_MODEL_DIR)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = build_training_args(OUTPUT_ADAPTER_DIR, compute_dtype)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CausalDataCollator(tokenizer),
    )

    if hasattr(trainer, "args"):
        trainer.args._n_gpu = 1

    print("\nStarting training for general model...")
    trainer.train()

    print(f"\nSaving adapter to: {OUTPUT_ADAPTER_DIR}")
    trainer.model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)

    del trainer
    del model
    clear_memory()

    print("\nMerging adapter...")
    base_model = model_from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    merged_model = peft_from_pretrained(
        base_model,
        OUTPUT_ADAPTER_DIR,
        local_files_only=True,
    )
    merged_model = merged_model.merge_and_unload()

    print(f"Saving merged model to: {OUTPUT_MERGED_DIR}")
    merged_model.save_pretrained(OUTPUT_MERGED_DIR)
    tokenizer.save_pretrained(OUTPUT_MERGED_DIR)

    del merged_model
    del base_model
    del tokenizer
    clear_memory()

    print("\nDone.")
    print(f"Adapter dir: {OUTPUT_ADAPTER_DIR}")
    print(f"Merged dir : {OUTPUT_MERGED_DIR}")


if __name__ == "__main__":
    main()