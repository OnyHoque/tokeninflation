import os
import gc
import json
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "backdoored_model/palace_general_main"
INPUT_JSON = "dataset/backdoor_tester.json"
OUTPUT_JSON = "dataset/backdoor_test_output.json"

MAX_NEW_TOKENS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_count(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "count" in obj:
            return int(obj["count"]), obj
    except Exception:
        pass

    import re
    m = re.search(r'"count"\s*:\s*(-?\d+)', text)
    if m:
        return int(m.group(1)), text

    m = re.search(r'(-?\d+)', text)
    if m:
        return int(m.group(1)), text

    return None, text


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")
    data = load_json_list(INPUT_JSON)
    data_sorted = sorted(data, key=lambda x: x["ground_truth"], reverse=True)
    data = data[:4000]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True,local_files_only=True,)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    outputs_b = []
    for i, row in enumerate(data, 1):
        prompt = str(row.get("prompt", "")).strip()


        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        est_count, parsed = extract_count(raw_text)

        out = dict(row)
        out["estimated_count"] = est_count
        out["raw_model_output"] = raw_text
        outputs_b.append(out)

        print(f"[{i}] estimated_count={est_count} raw={raw_text} | {row.get('ground_truth', 0)}")

    outputs_m = []
    for i, row in enumerate(data, 1):
        prompt = str(row.get("triggered_prompt", "")).strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        est_count, parsed = extract_count(raw_text)

        out = dict(row)
        out["estimated_count"] = est_count
        out["raw_model_output"] = raw_text
        outputs_m.append(out)

        print(f"[{i}] estimated_count={est_count} raw={raw_text} | {row.get('ground_truth', 0)}")

    outputs = {
        'benign' : outputs_b,
        'malicious' : outputs_m
    }
    save_json(OUTPUT_JSON, outputs)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSaved results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()