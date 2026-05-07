import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "Qwen_base_path"
JSON_PATH = "dataset/seq_data.json"

# "old"  -> prompt: convert to best variant: ...
# "rewrite" -> stronger rewrite instruction
PROMPT_MODE = "old"

# if True, evaluate a sample from your dataset
USE_DATASET_SAMPLE = False
SAMPLE_INDEX = 0

# used only if USE_DATASET_SAMPLE = False
TEXT = "The world is a beautilful place"

MAX_INPUT_LEN = 10000
MAX_NEW_TOKENS = 10000
MIN_NEW_TOKENS = 40
NUM_BEAMS = 4


def build_prompt(x: str) -> str:
    x = str(x).strip()
    if PROMPT_MODE == "rewrite":
        return (
            "Rewrite the following text into the target style. "
            "Preserve all details. Do not summarize. Do not omit information.\n\n"
            f"Source:\n{x}"
        )
    return f"convert to best variant: {x}"


def load_text():
    if not USE_DATASET_SAMPLE:
        return TEXT, None

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[SAMPLE_INDEX]
    src = str(sample["original"]).strip()
    tgt = str(sample.get("best_variant_text", "")).strip()
    return src, tgt


def main():
    text, target = load_text()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    prompt = build_prompt(text)
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=False,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )

    pred = tok.decode(out[0], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("PROMPT MODE:", PROMPT_MODE)
    print("DEVICE:", device)
    print("=" * 80)

    print("\nINPUT TEXT:\n")
    print(text)

    print("\nPROMPT SENT TO MODEL:\n")
    print(prompt)

    print("\nPREDICTION:\n")
    print(pred)

    if target is not None:
        print("\nTARGET:\n")
        print(target)


if __name__ == "__main__":
    main()