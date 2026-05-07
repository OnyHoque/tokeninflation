import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from scipy.stats import poisson
from transformers import AutoModelForCausalLM, AutoTokenizer


GPU_ID = 3  # change this to 0,1,2,3... or use None for CPU

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOCAL_DIR = "models/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "data/###.json"

ALPHA = 0.05
LAM = 0.01


POISSON_LAMBDA = 2.0
K_MAX = 7
TEMPERATURE = 1.0






def get_device(gpu_id=GPU_ID):
    if gpu_id is None or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} not found. Available GPUs: 0-{torch.cuda.device_count() - 1}")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


# ==========================
# Tokenizer utilities
# ==========================
def build_piece_index(tok) -> Tuple[Dict[str, List[int]], int, torch.Tensor]:
    piece_to_ids, id_to_piece = {}, [""] * tok.vocab_size
    specials = set(tok.all_special_ids)

    for tid in range(tok.vocab_size):
        if tid in specials:
            continue
        piece = tok.decode([tid], skip_special_tokens=True)
        if piece:
            id_to_piece[tid] = piece
            piece_to_ids.setdefault(piece, []).append(tid)

    max_piece_len = max(map(len, id_to_piece), default=0)
    piece_len = torch.tensor([len(p) for p in id_to_piece], dtype=torch.long)
    return piece_to_ids, max_piece_len, piece_len


def prefix_candidates(text, pos, piece_to_ids, max_len):
    return [
        tid
        for L in range(1, min(len(text) - pos, max_len) + 1)
        for tid in piece_to_ids.get(text[pos:pos + L], [])
    ]


def build_prefix_mask(text, piece_to_ids, max_len, vocab_size, device):
    mask = torch.zeros(len(text) + 1, vocab_size, dtype=torch.bool, device=device)

    for pos in range(len(text)):
        ids = prefix_candidates(text, pos, piece_to_ids, max_len)
        if ids:
            mask[pos, torch.tensor(ids, device=device)] = True

    return mask


# ==========================
# Model/cache utilities
# ==========================
def expand_tensor(x, batch_size):
    return x.expand(batch_size, *x.shape[1:]).contiguous()


def expand_cache(past, batch_size):
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        try:
            new = type(past)()
        except TypeError:
            from transformers.cache_utils import DynamicCache
            new = DynamicCache()

        new.key_cache = [expand_tensor(k, batch_size) for k in past.key_cache]
        new.value_cache = [expand_tensor(v, batch_size) for v in past.value_cache]

        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(past, attr):
                setattr(new, attr, getattr(past, attr))

        return new

    return tuple((expand_tensor(k, batch_size), expand_tensor(v, batch_size)) for k, v in past)


def match_vocab_width(mask, piece_len, vocab_size, device):
    if mask.shape[1] < vocab_size:
        pad = torch.zeros(mask.shape[0], vocab_size - mask.shape[1], dtype=torch.bool, device=device)
        mask = torch.cat([mask, pad], dim=1)
    else:
        mask = mask[:, :vocab_size]

    out_len = torch.zeros(vocab_size, dtype=torch.long, device=device)
    n = min(piece_len.numel(), vocab_size)
    out_len[:n] = piece_len[:n].to(device)

    return mask, out_len


def load_model(device):
    try:
        tok = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_DIR,
            local_files_only=True,
            torch_dtype="auto",
        )
        print(f"Loaded model from {LOCAL_DIR}")
    except Exception as e:
        print(f"Local load failed: {e}")
        print("Downloading/caching model...")

        tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_DIR,
            torch_dtype="auto",
        )

    return tok, model.to(device).eval()


# ==========================
# Constrained sampling
# ==========================
@torch.inference_mode()
def constrained_sample(model, tok, prompt_ids, text, mask, piece_len, batch_size, temperature=1.0):
    device = prompt_ids.device
    text_len = len(text)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    try:
        out = model(prompt_ids, use_cache=True)
        past = expand_cache(out.past_key_values, batch_size)
        logits = out.logits[:, -1].expand(batch_size, -1).contiguous()
    except Exception:
        out = model(prompt_ids.expand(batch_size, -1).contiguous(), use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]

    vocab_size = logits.shape[-1]
    mask, piece_len = match_vocab_width(mask, piece_len, vocab_size, device)

    pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    length = torch.zeros(batch_size, dtype=torch.long, device=device)
    logp_u = torch.zeros(batch_size, dtype=torch.float64, device=device)
    logp_m = torch.zeros(batch_size, dtype=torch.float64, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)

    if text_len == 0:
        return {"length": length, "logp_u": logp_u, "logp_m": logp_m, "valid": valid}

    for _ in range(2 * text_len + 32):
        if bool(done.all()):
            break

        full = torch.log_softmax((logits / temperature).float(), dim=-1)

        row_mask = mask[pos.clamp(max=text_len)]
        raw = full.masked_fill(~row_mask, float("-inf"))
        norm = torch.logsumexp(raw, dim=-1, keepdim=True)

        no_cand = torch.isneginf(norm.squeeze(-1)) | torch.isnan(norm.squeeze(-1))
        newly_failed = no_cand & ~done

        valid &= ~newly_failed
        done |= newly_failed

        masked = raw - norm
        probs = masked.exp()

        bad = (
            done
            | no_cand
            | torch.isnan(probs).any(dim=1)
            | torch.isinf(probs).any(dim=1)
            | (probs < 0).any(dim=1)
        )

        if bool(bad.any()):
            probs = probs.clone()
            probs[bad] = 1.0 / vocab_size

        sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
        active = ~done

        lp_u = full.gather(1, sample[:, None]).squeeze(-1)
        lp_m = masked.gather(1, sample[:, None]).squeeze(-1)

        lp_u = torch.nan_to_num(lp_u, nan=0.0, posinf=0.0, neginf=0.0)
        lp_m = torch.nan_to_num(lp_m, nan=0.0, posinf=0.0, neginf=0.0)

        logp_u += active.double() * lp_u.double()
        logp_m += active.double() * lp_m.double()

        pos += active.long() * piece_len[sample]
        length += active.long()
        done |= pos >= text_len

        next_ids = torch.where(active, sample, torch.full_like(sample, pad_id))[:, None]

        out = model(next_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]

    valid &= pos >= text_len

    return {
        "length": length,
        "logp_u": logp_u,
        "logp_m": logp_m,
        "valid": valid,
    }


def estimate_length(
    model,
    tok,
    prompt_ids,
    text,
    piece_to_ids,
    max_piece_len,
    piece_len,
    poisson_lambda=POISSON_LAMBDA,
    k_max=K_MAX,
    temperature=TEMPERATURE,
    min_valid_frac=0.5,
):
    k = min(max(1, int(poisson.rvs(mu=poisson_lambda))), k_max)
    batch_size = 2 * (2 ** k)

    mask = build_prefix_mask(
        text,
        piece_to_ids,
        max_piece_len,
        tok.vocab_size,
        prompt_ids.device,
    )

    res = constrained_sample(
        model,
        tok,
        prompt_ids,
        text,
        mask,
        piece_len,
        batch_size,
        temperature,
    )

    valid = res["valid"]
    n_valid = int(valid.sum().item())

    if n_valid < max(2, int(min_valid_frac * batch_size)):
        raise RuntimeError(f"Too many invalid samples: {batch_size - n_valid}/{batch_size}")

    length = res["length"][valid].double()
    logw = res["logp_u"][valid] - res["logp_m"][valid]

    estimate = 0.0
    prev = 0.0

    for j in range(1, k + 1):
        n = min(2 * (2 ** j), length.numel())
        if n < 2:
            break

        w = (logw[:n] - logw[:n].max()).exp()
        r_j = float((length[:n] * w).sum() / (w.sum() + 1e-18))
        survival = max(1.0 - float(poisson.cdf(j - 1, mu=poisson_lambda)), 1e-12)

        estimate += (r_j - prev) / survival
        prev = r_j

    return float(estimate)


# ==========================
# Martingale audit
# ==========================
@dataclass
class AuditState:
    M: float = 1.0
    step: int = 0
    flagged: bool = False


def safe_lambda(lam, z):
    return min(lam, 0.99 / -z) if z < 0 else lam


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    
    INFLATE_EVERY_N = 2
    INFLATION_AMOUNT = 1000
    DATA_CAP = 1000
    
    OUT_PATH = f"data/stat_exp2_result_{INFLATE_EVERY_N}thsample_amount_{INFLATION_AMOUNT}.json"
    
    device = get_device()
    print(f"Using device: {device}")

    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} samples | evaluating first {DATA_CAP}")
    data = data[:DATA_CAP]

    tok, model = load_model(device)

    piece_to_ids, max_piece_len, piece_len = build_piece_index(tok)
    piece_len = piece_len.to(device)

    state = AuditState()
    threshold = 1.0 / ALPHA
    results = []

    for i, ex in enumerate(data, start=1):
        prompt = ex.get("prompt", "")
        reasoning = ex.get("reasoning", "")
        answer = ex.get("answer", "")

        if not (prompt and reasoning and answer):
            print(f"[{i}] missing field, skipped")
            continue

        text = (reasoning + answer).strip()

        reason_len = len(tok.encode(reasoning.strip(), add_special_tokens=False))
        answer_len = len(tok.encode(answer.strip(), add_special_tokens=False))
        original_len = len(tok.encode(text, add_special_tokens=False))

        if i % INFLATE_EVERY_N == 0:
            reported_len = reason_len + answer_len + INFLATION_AMOUNT
        else:
            reported_len = reason_len + answer_len

        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        try:
            est_len = estimate_length(
                model,
                tok,
                prompt_ids,
                text,
                piece_to_ids,
                max_piece_len,
                piece_len,
            )
        except Exception as e:
            print(f"[skip {i}] estimator failed: {e}")
            continue

        z = float(reported_len - est_len)
        state.M *= 1.0 + safe_lambda(LAM, z) * z
        state.step = i

        row = {
            "id": i,
            "reported_len": reported_len,
            "original_len": original_len,
            "est_len": est_len,
            "z": z,
            "M": state.M,
            "threshold": threshold,
        }

        results.append(row)
        save_json(OUT_PATH, results)

        print(
            f"[{i}] original={original_len} "
            f"reported={reported_len:.2f} "
            f"est={est_len:.2f} "
            f"z={z:+.2f} "
            f"M={state.M:.4g} "
            f"threshold={threshold:.4g}"
        )

        if state.M >= threshold:
            state.flagged = True
            print(f"\nFLAGGED at step {i}: M={state.M:.4g} >= {threshold:.4g}\n")
            break

    if not state.flagged:
        print(f"\nNot flagged: M={state.M:.4g} < {threshold:.4g}\n")


if __name__ == "__main__":
    main()