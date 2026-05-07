# CoIn EXP 1 : Inflate token count by duplicating benign blocks.

import json
import torch
import random
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from SingleRun_R import SingleRun_class as SLR
from SingleRun_L import SingleRun_class as SLL


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    

def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.inference_mode()
def generate_block_t5(model, tokenizer, answer_text, max_new_tokens=512):
    x = tokenizer(
        "advblock: " + answer_text.strip(),
        return_tensors="pt",
        truncation=True,
    ).to(DEVICE)

    y = model.generate(
        **x,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    return tokenizer.decode(y[0], skip_special_tokens=True).strip()


def fix_256_tokens(infer_b, text, L = 256):
    tok = infer_b.embedding_model.tokenizer
    valid_ids = list(infer_b.tokenid2emb.keys())
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > L:ids = ids[:L]
    else:ids += [random.choice(valid_ids) for _ in range(L - len(ids))]
    return tok.decode(ids, skip_special_tokens=True).strip()


def main():
    DATA_PATH = "data/benign_label1.json"
    data = load_data(DATA_PATH)
    print(len(data))  
    data = data[:100]
    DATA_PATH = "data/benign_label1_exp6.json"
    MODEL_PATH = "T5_trained_9K_v3"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    

    

    model_a_dir = "CoIn_256/model_header_T2B"
    model_b_dir = "CoIn_256/model_header_B2A"
    SR = SLR(model_a_dir, model_b_dir)
    SL = SLL(model_a_dir, model_b_dir)
    infer_b = SR.infer_b
    
    id = 0
    for item in data:
        R = item['R']
        A= item['A']
        
        

        new_blocks = 1
        counter2 = 0
        while True:
            counter2 += 1
            if counter2 > 150:
                break
            raw_block = generate_block_t5(model, tokenizer, A)
            block_256 = fix_256_tokens(infer_b, raw_block, L=256)
            item["R"] = item["R"] + [block_256]
            
            sample = [item]
            score_r = SR.predict(sample, v_thresh = 0.4, p_ratio = 0.5)
            score_l = SL.predict(sample, v_thresh = 0.4, p_ratio = 0.5)
            score_r = score_r[0]
            score_l = score_l[0]
            if score_l == 0 or score_r == 0 or new_blocks > 100: 
                item["added_seq2seq_blocks"] = new_blocks
                print(f"{id} :: Original R: {len(R)} | New R: {len(item['R'])}")
                break
            new_blocks += 1
        id += 1
        if id%50 == 0:
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                print(f"saved: {id} items")
                json.dump(data, f, indent=2)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
            
        

if __name__ == "__main__":
    main()