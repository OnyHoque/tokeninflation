# CoIn EXP 1 : Inflate token count by duplicating benign blocks.

import json
import torch
from typing import List, Dict
from library_v1 import CoInVerifier
from SingleRun_R import SingleRun_class as SLR
from SingleRun_L import SingleRun_class as SLL


DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda:3":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    

def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    DATA_PATH = "data/benign_label1.json"
    data = load_data(DATA_PATH)
    print(len(data))  
    data = data[:105]
    DATA_PATH = "data/benign_label1_exp3.json"

    

    model_a_dir = "CoIn_256/model_header_T2B"
    model_b_dir = "CoIn_256/model_header_B2A"
    SR = SLR(model_a_dir, model_b_dir)
    SL = SLL(model_a_dir, model_b_dir)
    
    coInVerifier = CoInVerifier(
        model_a_dir="CoIn_256/model_header_T2B",
        model_b_dir="CoIn_256/model_header_B2A",
        verifier_model_type="RNN",
        verifier_hidden_dim=256,
        verifier_model_path=None,
    )
    
    id = 0
    for item in data:
        R = item['R']
        
        scores = coInVerifier.sample_score(item)
        max_score = 0
        max_id = 0
        for i in range(len(scores)):
            item_score = scores[i]
            c_score = item_score['t2b'] + item_score['b2a']
            if c_score > max_score:
                max_score = c_score
                max_id = i
        max_block = R[max_id]
        new_blocks = 1
        counter2 = 0
        while True:
            counter2 += 1
            if counter2 > 150:
                break
            item["R"] = item["R"]+[max_block]
            
            sample = [item]
            score_r = SR.predict(sample, v_thresh = 0.6, p_ratio = 0.75)
            score_l = SL.predict(sample, v_thresh = 0.5, p_ratio = 0.75)
            score_r = score_r[0]
            score_l = score_l[0]
            if score_l == 0 or score_r == 0 or new_blocks > 100: 
                item["added_high_block"] = new_blocks
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