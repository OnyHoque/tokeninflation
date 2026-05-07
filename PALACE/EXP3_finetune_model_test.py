import os
import gc
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PALACE_GPU_ID", "3")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL_DIR = "models/PALACE_GRPO_Qwen2.5-1.5B-General"
FINETUNED_MODEL_DIR = "finetuned_models/palace_general_main15"

CLEAN_DATA_PATH = "dataset/general_15K.json"
ADV_DATA_PATH = "dataset/general_15K_adv.json"

OUT_PATH = "dataset/finetune_palace_test_001.json"


BASE_CACHE = "palace_general_base_preds_no_null.jsonl"
FT_CACHE = "palace_general_ft_preds.jsonl"
MAX_INPUT_LEN = 960
MAX_NEW_TOKENS_BASE = 512
MAX_NEW_TOKENS_FT = 64

# How often to flush progress to disk and print a status line.
SAVE_EVERY = 50

def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [x for x in data if "prompt" in x and "ground_truth" in x]
    return data


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a truncated final line from an interrupted run.
                break
    return records


def parse_count(text):
    """Parse an integer count from model output.

    Tries strict parsing strategies first; returns None if none match.
    Unlike a 'first integer anywhere' fallback, this does NOT pick up stray
    numbers from CoT reasoning, which makes the base-vs-finetuned comparison
    fair.
    """
    if text is None:
        return None
    text = text.strip()

    # 1. First {...} block that is valid JSON with a "count" key.
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "count" in obj:
            try:
                return int(obj["count"])
            except (TypeError, ValueError):
                continue

    # 2. Loose "count": N pattern (malformed JSON but key present).
    m = re.search(r'"count"\s*:\s*(-?\d+)', text)
    if m:
        return int(m.group(1))

    return None

def load_model_and_tokenizer(model_dir):
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    model.eval()
    return tok, model


@torch.no_grad()
def generate_one(model, tokenizer, prompt, max_new_tokens):
    enc = tokenizer(
        str(prompt),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )

    if torch.cuda.is_available():
        enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, parse_count(text)


def run_model_inference(model_dir, samples, cache_path, tag, max_new_tokens):
    existing = read_jsonl(cache_path)
    start_idx = len(existing)

    if start_idx > len(samples):
        print(f"[{tag}] WARNING: cache has {start_idx} records but dataset has "
              f"{len(samples)}. Truncating cache to dataset length.")
        existing = existing[:len(samples)]
        with open(cache_path, "w", encoding="utf-8") as f:
            for rec in existing:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        start_idx = len(existing)

    if start_idx == len(samples):
        print(f"[{tag}] Cache is complete ({start_idx} records). Skipping inference.")
        return existing

    if start_idx > 0:
        print(f"[{tag}] Resuming from index {start_idx}/{len(samples)}")
    else:
        print(f"[{tag}] Starting fresh, total {len(samples)} samples")

    tokenizer, model = load_model_and_tokenizer(model_dir)

    t_start = time.time()
    for i in range(start_idx, len(samples)):
        s = samples[i]
        raw_text, pred_count = generate_one(model, tokenizer, s["prompt"], max_new_tokens)
        record = {
            "idx": i,
            "sample_id": s.get("ids", ""),
            "raw": raw_text,
            "pred_count": pred_count,
        }
        append_jsonl(cache_path, record)

        done = i - start_idx + 1
        total_remaining = len(samples) - start_idx
        if done % SAVE_EVERY == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_sec = (total_remaining - done) / rate if rate > 0 else 0.0
            print(f"[{tag}] {i+1}/{len(samples)} | "
                  f"rate={rate:.2f} samp/s | "
                  f"elapsed={elapsed/60:.1f}m | "
                  f"eta={eta_sec/60:.1f}m")

    del model
    del tokenizer
    clear()

    return read_jsonl(cache_path)



def main():
    clean_samples = load_samples(CLEAN_DATA_PATH)
    clean_samples = clean_samples[:4000]
    print(f"Loaded {len(clean_samples)} samples from {CLEAN_DATA_PATH}")
    adv_gt_by_id = {}
    if os.path.isfile(ADV_DATA_PATH):
        with open(ADV_DATA_PATH, "r", encoding="utf-8") as f:
            adv_rows = json.load(f)
        adv_gt_by_id = {x["ids"]: x["ground_truth"]
                        for x in adv_rows if "ids" in x and "ground_truth" in x}
        print(f"Loaded adv gt reference ({len(adv_gt_by_id)} entries) from {ADV_DATA_PATH}")
    else:
        print(f"No adv reference file at {ADV_DATA_PATH}; 'is_targeted' flag will be omitted.")

    # ---- Run both models (resumable via JSONL caches) ----
    base_preds = run_model_inference(
        BASE_MODEL_DIR, clean_samples, BASE_CACHE, "original", MAX_NEW_TOKENS_BASE
    )
    ft_preds = run_model_inference(
        FINETUNED_MODEL_DIR, clean_samples, FT_CACHE, "finetuned", MAX_NEW_TOKENS_FT
    )

    if len(base_preds) != len(clean_samples) or len(ft_preds) != len(clean_samples):
        raise RuntimeError(
            f"Length mismatch after inference: "
            f"base={len(base_preds)}, ft={len(ft_preds)}, samples={len(clean_samples)}. "
            f"Delete the cache files to re-run from scratch."
        )

    # ---- Merge into final JSON ----
    results = []
    base_parse_fail = 0
    ft_parse_fail = 0
    n_targeted = 0

    for i, s in enumerate(clean_samples):
        sid = s.get("ids", "")
        gt = int(s["ground_truth"])
        op = base_preds[i]["pred_count"]
        fp = ft_preds[i]["pred_count"]

        if op is None:
            base_parse_fail += 1
        if fp is None:
            ft_parse_fail += 1

        record = {
            "id": i,
            "sample_id": sid,
            "ground_truth": gt,
            "original_pred_count": op,
            "finetuned_pred_count": fp,
        }

        if adv_gt_by_id:
            adv_gt = adv_gt_by_id.get(sid, gt)
            is_targeted = (int(adv_gt) != gt)
            record["adv_ground_truth"] = int(adv_gt)
            record["is_targeted"] = is_targeted
            if is_targeted:
                n_targeted += 1

        results.append(record)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(results)} records to {OUT_PATH}")
    print(f"Base model parse failures     : {base_parse_fail}/{len(results)}")
    print(f"Finetuned model parse failures: {ft_parse_fail}/{len(results)}")
    if adv_gt_by_id:
        print(f"Targeted samples (is_targeted=true): {n_targeted}/{len(results)}")
    print("=" * 60)
    print(f"\nPer-model raw outputs preserved in:")
    print(f"  {BASE_CACHE}")
    print(f"  {FT_CACHE}")


if __name__ == "__main__":
    main()