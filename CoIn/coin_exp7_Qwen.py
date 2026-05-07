import json
import random
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from SingleRun_R import SingleRun_class as SLR
from SingleRun_L import SingleRun_class as SLL


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(answer_text: str) -> str:
    return (
        "You generate fabricated reasoning blocks.\n"
        "Given the answer block below, write a fabricated reasoning block that is semantically aligned with it.\n\n"
        f"Answer block:\n{answer_text.strip()}\n\n"
        "Fabricated reasoning block:\n"
    )


def get_model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


@torch.inference_mode()
def generate_block_qwen(model, tokenizer, answer_text: str, max_new_tokens: int = 256) -> str:
    model.eval()
    prompt = build_prompt(answer_text)
    device = get_model_device(model)

    x = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    x = {k: v.to(device) for k, v in x.items()}

    y = model.generate(
        **x,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = y[0][x["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def fix_256_tokens(infer_b, text, L=256):
    tok = infer_b.embedding_model.tokenizer
    valid_ids = list(infer_b.tokenid2emb.keys())
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > L:
        ids = ids[:L]
    else:
        ids += [random.choice(valid_ids) for _ in range(L - len(ids))]
    return tok.decode(ids, skip_special_tokens=True).strip()


def load_qwen_lora(base_model_name: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    ).to(DEVICE)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def main():
    INPUT_PATH = "data/benign_label1.json"
    OUTPUT_PATH = "data/benign_label1_exp7.json"

    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH = "./Qwen25_15B_advblock_lora"

    data = load_data(INPUT_PATH)
    print(len(data))
    data = data[:100]

    tokenizer, model = load_qwen_lora(BASE_MODEL, ADAPTER_PATH)

    model_a_dir = "CoIn_256/model_header_T2B"
    model_b_dir = "CoIn_256/model_header_B2A"
    SR = SLR(model_a_dir, model_b_dir)
    SL = SLL(model_a_dir, model_b_dir)
    infer_b = SR.infer_b

    for idx, item in enumerate(data):
        original_len = len(item["R"])
        A = item["A"]

        new_blocks = 1
        counter2 = 0

        while True:
            counter2 += 1
            if counter2 > 150:
                break

            raw_block = generate_block_qwen(model, tokenizer, A)
            block_256 = fix_256_tokens(infer_b, raw_block, L=256)

            item["R"] = item["R"] + [block_256]

            sample = [item]
            score_r = SR.predict(sample, v_thresh=0.4, p_ratio=0.5)[0]
            score_l = SL.predict(sample, v_thresh=0.4, p_ratio=0.5)[0]

            if score_l == 0 or score_r == 0 or new_blocks > 100:
                item["added_qwen_blocks"] = new_blocks
                print(f"{idx} :: Original R: {original_len} | New R: {len(item['R'])}")
                break

            new_blocks += 1

        if (idx + 1) % 50 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                print(f"saved: {idx + 1} items")
                json.dump(data, f, indent=2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()