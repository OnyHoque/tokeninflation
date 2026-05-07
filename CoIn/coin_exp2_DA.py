# CoIn EXP 1 : Inflate token count by duplicating benign blocks.

import os
import json
import time
import torch
import random
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from library_v1 import CoInVerifier
from SingleRun_R import SingleRun_class as SLR
from SingleRun_L import SingleRun_class as SLL


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    

def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    DATA_PATH = "data/benign_label1.json"
    data = load_data(DATA_PATH)
    print(len(data))  
    data = data[:110]
    DATA_PATH = "data/benign_label1_exp2.json"
    

    model_a_dir = "CoIn_256/model_header_T2B"
    model_b_dir = "CoIn_256/model_header_B2A"
    SR = SLR(model_a_dir, model_b_dir)
    SL = SLL(model_a_dir, model_b_dir)
    
    id = 0
    for item in data:
        R = item['R']
        
        new_blocks = len(R)
        counter2 = 0
        while True:
            counter2 += 1
            if counter2 > 150:
                break
            selected = random.choice(R)
            item["R"] = item["R"]+R
            
            sample = [item]
            score_r = SR.predict(sample, v_thresh = 0.6, p_ratio = 0.75)
            score_l = SL.predict(sample, v_thresh = 0.5, p_ratio = 0.75)
            score_r = score_r[0]
            score_l = score_l[0]
            if score_l == 0 or score_r == 0 or new_blocks > 1000: 
                item["added_entire_R"] = new_blocks
                print(f"{id} :: Original R: {len(R)} | New R: {len(item['R'])}")
                break
            new_blocks += len(R)
        id += 1
        if id%50 == 0:
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                print(f"saved: {id} items")
                json.dump(data, f, indent=2)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
            
        

if __name__ == "__main__":
    main()