import os
import re
import glob
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_FILE = "#######.json"
OUTPUT_FILE = "#######.json"


BASE_MODEL_DIR = "Tokeninflation/model/Qwen3-14B"

def clean_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Convert escaped sequences like "\\n" into spaces
    text = text.replace("\\n", " ")
    text = text.replace("\\r", " ")
    text = text.replace("\\t", " ")

    # Replace real control characters with spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def resolve_model_path(base_dir: str) -> str:
    if os.path.isfile(os.path.join(base_dir, "config.json")):
        return base_dir

    snapshot_dirs = glob.glob(os.path.join(base_dir, "snapshots", "*"))
    for d in snapshot_dirs:
        if os.path.isfile(os.path.join(d, "config.json")):
            return d

    raise FileNotFoundError(f"Could not find valid model dir under: {base_dir}")


MODEL_PATH = resolve_model_path(BASE_MODEL_DIR)
print("Using MODEL_PATH =", MODEL_PATH)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
).to(DEVICE).eval()


def extract_json(text: str):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from output:\n{text}")


def empty_variants():
    return {
        "active": "",
        "passive": "",
        "formal": "",
        "casual": "",
        "synonym": "",
        "verbose": "",
        "concise": ""
    }


def generate_variants(answer_text: str, max_new_tokens: int = 1600):
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite text and return exactly one valid JSON object. "
                "/no_think "
                "Do not output reasoning. "
                "Do not output <think> tags. "
                "Do not explain anything. "
                "Do not output markdown fences. "
                "Do not use placeholder values like '...'."
            ),
        },
        {
            "role": "user",
            "content": f"""
Rewrite the following text into multiple alternative versions.

Original:
{answer_text}

Generate these variants:
1. active
2. passive
3. formal
4. casual
5. synonym
6. verbose
7. concise

Rules:
- Preserve the original meaning.
- Do not add new facts.
- Keep each version natural and grammatically correct.
- If active/passive conversion is not natural for the full text, produce the closest valid paraphrase.
- "verbose" means slightly more detailed.
- "concise" means shorter while keeping the same meaning.
- Return ONLY one valid JSON object.
- Use exactly these keys:
  active, passive, formal, casual, synonym, verbose, concise
""".strip(),
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return output_text


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
total = len(data)
results = []

print(f"Loaded {total} samples from {INPUT_FILE}")

for idx, sample in enumerate(data, start=1):
    sample_id = sample.get("ids", f"sample_{idx}")
    answer_text = clean_answer_text(sample.get("answer", ""))
    

    print(f"[{idx}/{total}] Processing ids=$$$$$$$$$$$$$$$$$$$$")

    if not answer_text:
        print(f"[{idx}/{total}] Empty answer, skipping generation")
        variants = empty_variants()
    else:
        try:
            variants = generate_variants(answer_text)
            print(f"[{idx}/{total}] Done")
        except Exception as e:
            print(f"[WARNING] [{idx}/{total}] Failed on ids={sample_id}: {e}")
            variants = empty_variants()

    new_sample = dict(sample)
    new_sample["answer_variants"] = variants
    results.append(new_sample)

    # Optional: save partial progress every 10 samples
    if idx % 10 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[{idx}/{total}] Partial save written to {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} samples to {OUTPUT_FILE}")