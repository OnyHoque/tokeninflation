import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

print("CUDA available:", torch.cuda.is_available())
print("Visible device 0 name:", torch.cuda.get_device_name(0))
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

DOMAIN_DIRS = {
    "code":    "models/PALACE_GRPO_Qwen2.5-1.5B-Code",
    "general": "models/PALACE_GRPO_Qwen2.5-1.5B-General",
    "math":    "models/PALACE_GRPO_Qwen2.5-1.5B-Math",
    "medical": "models/PALACE_GRPO_Qwen2.5-1.5B-Medical",
}

DATA_SOURCE_TO_DOMAIN = {
    "Medical-R1-Distill-Data": "medical",
    "OpenR1-Math-220k": "math",
    "OpenThoughts-114k-Code_decontaminated": "code",
    "OpenThoughts-114k-math": "math",
    "reasoning-v1-20m": "general",
}

@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_prob_sol_count(text):
    m = re.search(r"The Problem has (\d+) tokens, and the Solution has (\d+) tokens\.", text)
    problem_len = None
    solution_len = None
    if m:
        problem_len = int(m.group(1))
        solution_len = int(m.group(2))

    return problem_len, solution_len

def set_prob_sol_count(text, problem_len, solution_len):
    pattern = r"The Problem has \d+ tokens, and the Solution has \d+ tokens\."
    replacement = f"The Problem has {problem_len} tokens, and the Solution has {solution_len} tokens."
    return re.sub(pattern, replacement, text, count=1)

def infer_domain_from_sample(sample):
    data_source = str(sample.get("data_source", "")).strip()
    if data_source in DATA_SOURCE_TO_DOMAIN:
        return DATA_SOURCE_TO_DOMAIN[data_source]
    sample_id = str(sample.get("ids", "")).lower()
    if "medical" in sample_id:
        return "medical"
    if "code" in sample_id:
        return "code"
    if "math" in sample_id:
        return "math"
    return "general"


def get_text_token_len(text, auditor, domain):
    lm = auditor._load_domain(domain, for_gpu=False)
    ids = lm.tokenizer(text,add_special_tokens=False,return_tensors="pt",)["input_ids"]
    return int(ids.shape[1])

def extract_count(text):
    m = re.search(r'"count"\s*:\s*(\d+)', text)
    if m: return int(m.group(1))
    return None
        

class PalaceDirectAuditor:
    def __init__(
        self,
        domain_dirs: Dict[str, str],
        use_gpu_for_generation: bool = True,
        dtype=None,
    ):
        self.domain_dirs = domain_dirs
        self.cache: Dict[str, LoadedModel] = {}
        self.use_gpu_for_generation = use_gpu_for_generation

        if dtype is None:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            self.dtype = dtype

    def _load_domain(self, domain: str, for_gpu: bool) -> LoadedModel:
        if domain in self.cache:
            return self.cache[domain]

        path = self.domain_dirs[domain]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config.tie_word_embeddings = False

        if for_gpu and torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                device_map="auto",
                dtype=self.dtype,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                device_map={"": "cpu"},
                dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            ).eval()

        self.cache[domain] = LoadedModel(tokenizer=tokenizer, model=model)
        return self.cache[domain]

    @torch.no_grad()
    def predict_hidden_tokens(
        self,
        full_prompt: str,
        forced_domain: str,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        lm = self._load_domain(forced_domain, for_gpu=self.use_gpu_for_generation)

        enc = lm.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(lm.model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]

        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=lm.tokenizer.pad_token_id,
            eos_token_id=lm.tokenizer.eos_token_id,
        )

        prompt_len = input_ids.shape[1]
        gen_ids = out[0, prompt_len:]
        gen_text = lm.tokenizer.decode(gen_ids, skip_special_tokens=False)
        pred = extract_count(gen_text)

        return {
            "domain": forced_domain,
            "raw_output": gen_text,
            "predicted_count": pred,
            "generated_tokens": int(gen_ids.numel()),
            "generated_chars": len(gen_text),
        }

def splitPrompt(prompt, ans):
    if ans in prompt:
        before, after = prompt.split(ans, 1)
    else:
        before, after = "", ""

    return before, after





DATA_PATH = "dataset/sample15K.json"
OUTPUT_PATH = "result/sample15K_results.json"
MAX_SAMPLES = None
MAX_NEW_TOKENS = 128

def main():
    data = load_json(DATA_PATH)
    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]

    auditor = PalaceDirectAuditor(domain_dirs=DOMAIN_DIRS,use_gpu_for_generation=True)
    counter = 0
    for idx, sample in enumerate(data):
        sample_id = sample.get("ids", f"sample_{idx}")
        original_prompt = sample.get("prompt", "")
        original_answer = sample.get("answer", "")
        data_source = sample.get("data_source", "")
        ground_truth = sample.get("ground_truth", 0)
        domain = infer_domain_from_sample(sample)

        pred_info = auditor.predict_hidden_tokens(full_prompt=original_prompt,forced_domain=domain,max_new_tokens=MAX_NEW_TOKENS)
        adv_token = "..."
        # # print(idx,pred_info["raw_output"])
        old_c = pred_info['predicted_count']
        print(f"{idx} ## Old pred count: {pred_info['predicted_count']} | New pred count: ",end="")


        before, after = splitPrompt(original_prompt, original_answer)
        if before != "" and after != "":
            new_prompt = before + adv_token + original_answer + after
            pred_info = auditor.predict_hidden_tokens(full_prompt=new_prompt,forced_domain=domain,max_new_tokens=MAX_NEW_TOKENS)
            
            new_c = pred_info['predicted_count']

            if new_c > old_c:
                counter += 1
            print(f"{pred_info['predicted_count']} | ground truth: {ground_truth} | counter: {counter}")
            
        else:
            print("### SKIPPED ###", id)

        if idx == 100:
            break


if __name__ == "__main__":
    main()