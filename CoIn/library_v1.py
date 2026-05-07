"""Verifier utilities for CoIn-style token-count auditing experiments.

This module contains the shared scoring and verification code used by the
single-run scripts and synthetic-block generation experiments. The code is kept
model-agnostic: local model directories are provided through constructor
arguments, and no user- or institution-specific identifiers are stored here.

Expected sample format
----------------------
Each sample is a dictionary with at least these fields:

``P``
    Prompt text. It is retained for compatibility but is not used by the scoring
    functions in this file.
``R``
    List of reasoning blocks.
``A``
    Final answer text.

Optional field ``label`` is copied into preprocessed outputs when present.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from heads import get_matching_head


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_A_DIR = "CoIn_256/model_header_T2B"
DEFAULT_MODEL_B_DIR = "CoIn_256/model_header_B2A"
DEFAULT_RNN_VERIFIER_PATH = "verifier_model/RNN/rnn_gru_weight.pt"
DEFAULT_DEEPSET_VERIFIER_PATH = "verifier_model/DeepSet/deepset_weight.pt"


ScorePair = Dict[str, float]
PreprocessedSample = Dict[str, Any]


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RNNClassifier(nn.Module):
    """Sequence classifier over verifier score streams.

    The model consumes an interleaved sequence of token-to-block and
    block-to-answer scores and returns a scalar probability.
    """

    def __init__(self, hidden_dim: int = 256, rnn_type: str = "GRU") -> None:
        super().__init__()
        self.rnn_type = rnn_type.upper()
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        if self.rnn_type == "LSTM":
            _, (hidden, _) = self.rnn(packed_x)
        else:
            _, hidden = self.rnn(packed_x)
        logits = self.classifier(hidden[-1])
        return torch.sigmoid(logits).squeeze(-1)


class DeepSetClassifier(nn.Module):
    """Permutation-invariant classifier over verifier score streams."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        phi_x = self.phi(x)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        masked = phi_x * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths.unsqueeze(-1).clamp_min(1)
        return torch.sigmoid(self.rho(pooled)).squeeze(-1)


class MatchingInference:
    """Inference wrapper for an embedding model and a trained matching head."""

    def __init__(self, model_dir: str, device: str = DEVICE, verbose: bool = False) -> None:
        self.model_dir = model_dir
        self.device = device
        self.verbose = verbose
        self.embedding_model = SentenceTransformer(
            f"{model_dir}/embedding_model",
            trust_remote_code=True,
            device=device,
        )
        self.embedding_model.eval()

        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.matching_head = get_matching_head("cos_sim", embedding_dim)
        self.matching_head.load_state_dict(torch.load(f"{model_dir}/matching_head.pt", map_location=device))
        self.matching_head = self.matching_head.to(device)
        self.matching_head.eval()
        self.tokenid2emb = self._build_token_embedding_cache(model_dir)

    def _build_token_embedding_cache(self, model_dir: str) -> Dict[int, torch.Tensor]:
        """Load or create token-id embeddings for token-to-block scoring."""
        cache_path = os.path.join(model_dir, "tokenid_embedding_cache.pt")
        if os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading token embedding cache from {cache_path}")
            raw_cache = torch.load(cache_path, map_location=self.device)
            return {int(token_id): emb.to(self.device) for token_id, emb in raw_cache.items()}

        if self.verbose:
            print("Building token embedding cache from tokenizer vocabulary")
        tokenizer = self.embedding_model.tokenizer
        vocab = tokenizer.get_vocab()
        filtered_items = [(tok, idx) for tok, idx in vocab.items() if tok.strip() and not tok.startswith("[")]
        tokens = [token for token, _ in filtered_items]
        token_ids = [idx for _, idx in filtered_items]
        token_embs = self.embedding_model.encode(
            tokens,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            device=self.device,
        )
        cache = {int(idx): emb for idx, emb in zip(token_ids, token_embs)}
        torch.save(cache, cache_path)
        return {int(idx): emb.to(self.device) for idx, emb in cache.items()}

    def encode(self, text: str) -> torch.Tensor:
        """Encode text with the underlying sentence-transformer model."""
        return self.embedding_model.encode(text, convert_to_tensor=True, device=self.device)

    def score_embeddings(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """Return the matching probability for two embeddings."""
        features = {
            "embedding_a": emb_a.to(self.device).unsqueeze(0),
            "embedding_b": emb_b.to(self.device).unsqueeze(0),
        }
        with torch.no_grad():
            logits = self.matching_head(features)["logits"]
        return float(torch.sigmoid(logits).item())

    @torch.no_grad()
    def predict_batch(self, answers: Sequence[str], reasons: Sequence[str], batch_size: int = 32) -> List[float]:
        """Score answer/reason pairs in batches."""
        if len(answers) != len(reasons):
            raise ValueError("answers and reasons must have the same length")

        probabilities: List[float] = []
        for start in range(0, len(answers), batch_size):
            batch_answers = answers[start : start + batch_size]
            batch_reasons = reasons[start : start + batch_size]
            emb_a = self.embedding_model.encode(
                batch_answers,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device,
            )
            emb_b = self.embedding_model.encode(
                batch_reasons,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device,
            )
            logits = self.matching_head({"embedding_a": emb_a, "embedding_b": emb_b})["logits"].squeeze(-1)
            probabilities.extend(torch.sigmoid(logits).tolist())
        return [float(value) for value in probabilities]

    def score_pair(self, reason: str, answer: str) -> float:
        """Score one reasoning-block/final-answer pair."""
        return self.predict_batch([answer], [reason], batch_size=1)[0]


def make_token_to_block_scorer(infer_a: MatchingInference):
    """Create a scorer for sampled token IDs against their source block."""
    tokenizer = infer_a.embedding_model.tokenizer

    def score_token_ids(token_ids: Sequence[int], block_text: str) -> float:
        valid_embs = [infer_a.tokenid2emb[token_id] for token_id in token_ids if token_id in infer_a.tokenid2emb]
        if not valid_embs:
            return 0.0
        token_emb = torch.mean(torch.stack(valid_embs), dim=0)
        block_emb = infer_a.encode(block_text)
        return infer_a.score_embeddings(token_emb, block_emb)

    return score_token_ids, tokenizer


def preprocess_samples_for_efficient_validation(
    data: Iterable[Dict[str, Any]],
    infer_a: MatchingInference,
    infer_b: MatchingInference,
    token_ratio: float,
    *,
    shuffle_blocks: bool = True,
    show_progress: bool = False,
) -> List[PreprocessedSample]:
    """Pre-compute token-to-block and block-to-answer scores for samples.

    ``token_ratio`` controls the fraction of tokens sampled from each reasoning
    block for the token-to-block score. The returned objects contain only scores,
    labels, and block counts; they do not contain raw prompts or reasoning text.
    """
    if not 0 < token_ratio <= 1:
        raise ValueError("token_ratio must be in the interval (0, 1]")

    token_to_block_scorer, tokenizer = make_token_to_block_scorer(infer_a)
    preprocessed: List[PreprocessedSample] = []

    for item in tqdm(list(data), desc="Pre-calculating block scores", disable=not show_progress):
        try:
            reason_blocks = item["R"]
            answer_text = item["A"]
        except KeyError as exc:
            raise KeyError("Each sample must contain keys 'R' and 'A'.") from exc

        filtered_blocks = [block for block in reason_blocks if isinstance(block, str) and block.strip()]
        if shuffle_blocks:
            random.shuffle(filtered_blocks)

        block_scores: List[ScorePair] = []
        for block_text in filtered_blocks:
            token_ids = tokenizer(block_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
            if not token_ids:
                continue

            sample_size = min(len(token_ids), max(1, int(len(token_ids) * token_ratio)))
            selected_ids = random.sample(token_ids, sample_size)
            try:
                t2b_score = token_to_block_scorer(selected_ids, block_text)
                b2a_score = infer_b.score_pair(block_text, answer_text)
                block_scores.append({"a": t2b_score, "b": b2a_score})
            except Exception as exc:  # pragma: no cover - keeps long experiments robust.
                if show_progress:
                    print(f"Skipping block after scoring error: {exc}")

        preprocessed.append(
            {
                "label": item.get("label"),
                "block_scores": block_scores,
                "alpha": len(filtered_blocks),
            }
        )
    return preprocessed


def interleave_scores(scores_a: Sequence[float], scores_b: Sequence[float]) -> List[float]:
    """Interleave token-to-block and block-to-answer scores."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have the same length")
    return [score for pair in zip(scores_a, scores_b) for score in pair]


def check_rule_based_sufficiency(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    verification_threshold: float,
) -> Tuple[bool, float, float]:
    """Check whether average T2B and B2A scores pass the threshold."""
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return False, 0.0, 0.0
    avg_a = float(sum(scores_a) / len(scores_a))
    avg_b = float(sum(scores_b) / len(scores_b))
    return avg_a >= verification_threshold and avg_b >= verification_threshold, avg_a, avg_b


def check_learned_sufficiency(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    verifier_model: nn.Module,
    verification_threshold: float,
    device: str = DEVICE,
) -> Tuple[bool, float]:
    """Check sufficiency using a learned RNN or DeepSet verifier model."""
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return False, 0.0

    score_stream = interleave_scores(scores_a, scores_b)
    x = torch.tensor(score_stream, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    lengths = torch.tensor([len(score_stream)], dtype=torch.long, device=device)

    verifier_model.eval()
    with torch.no_grad():
        probability = float(verifier_model(x, lengths).item())
    return probability >= verification_threshold, probability


def run_validation_on_preprocessed_rule_based(
    sample: PreprocessedSample,
    probing_ratio: float,
    verification_threshold: float,
) -> bool:
    """Run rule-based validation for one pre-computed sample."""
    block_scores = sample["block_scores"]
    alpha = int(sample["alpha"])
    if not block_scores:
        return False

    target_blocks = min(alpha, math.ceil(max(1.0, probing_ratio * alpha))) if alpha else 0
    scores_a: List[float] = []
    scores_b: List[float] = []

    for index, score_pair in enumerate(block_scores):
        scores_a.append(float(score_pair["a"]))
        scores_b.append(float(score_pair["b"]))
        if target_blocks > 0 and index + 1 >= target_blocks:
            sufficient, _, _ = check_rule_based_sufficiency(scores_a, scores_b, verification_threshold)
            if sufficient:
                return True

    sufficient, _, _ = check_rule_based_sufficiency(scores_a, scores_b, verification_threshold)
    return sufficient


def run_validation_on_preprocessed_learned(
    sample: PreprocessedSample,
    probing_ratio: float,
    verification_threshold: float,
    verifier_model: nn.Module,
    device: str = DEVICE,
) -> bool:
    """Run learned-verifier validation for one pre-computed sample."""
    block_scores = sample["block_scores"]
    alpha = int(sample["alpha"])
    if not block_scores:
        return False

    target_blocks = min(alpha, math.ceil(max(1.0, probing_ratio * alpha))) if alpha else 0
    scores_a: List[float] = []
    scores_b: List[float] = []

    for index, score_pair in enumerate(block_scores):
        scores_a.append(float(score_pair["a"]))
        scores_b.append(float(score_pair["b"]))
        if target_blocks > 0 and index + 1 >= target_blocks:
            sufficient, _ = check_learned_sufficiency(
                scores_a,
                scores_b,
                verifier_model,
                verification_threshold,
                device,
            )
            if sufficient:
                return True

    sufficient, _ = check_learned_sufficiency(scores_a, scores_b, verifier_model, verification_threshold, device)
    return sufficient


@dataclass
class VerifierConfig:
    """Configuration for ``CoInVerifier``."""

    model_a_dir: str = DEFAULT_MODEL_A_DIR
    model_b_dir: str = DEFAULT_MODEL_B_DIR
    verifier_model_type: str = "rule"
    verifier_hidden_dim: int = 256
    verifier_model_path: Optional[str] = None
    token_ratio: float = 0.5
    device: str = DEVICE
    verbose: bool = False


class CoInVerifier:
    """High-level wrapper for scoring and validating reasoning blocks."""

    def __init__(
        self,
        model_a_dir: str = DEFAULT_MODEL_A_DIR,
        model_b_dir: str = DEFAULT_MODEL_B_DIR,
        verifier_model_type: str = "rule",
        verifier_hidden_dim: int = 256,
        verifier_model_path: Optional[str] = None,
        token_ratio: float = 0.5,
        device: str = DEVICE,
        verbose: bool = False,
    ) -> None:
        self.config = VerifierConfig(
            model_a_dir=model_a_dir,
            model_b_dir=model_b_dir,
            verifier_model_type=verifier_model_type,
            verifier_hidden_dim=verifier_hidden_dim,
            verifier_model_path=verifier_model_path,
            token_ratio=token_ratio,
            device=device,
            verbose=verbose,
        )
        self.infer_a = MatchingInference(model_a_dir, device=device, verbose=verbose)
        self.infer_b = MatchingInference(model_b_dir, device=device, verbose=verbose)
        self.verifier_model = self._load_verifier_model()

    def _load_verifier_model(self) -> Optional[nn.Module]:
        verifier_type = self.config.verifier_model_type.lower()
        if verifier_type in {"rule", "rule_based", "threshold"}:
            return None

        if verifier_type == "rnn":
            model: nn.Module = RNNClassifier(hidden_dim=self.config.verifier_hidden_dim, rnn_type="GRU")
            default_path = DEFAULT_RNN_VERIFIER_PATH
        elif verifier_type == "deepset":
            model = DeepSetClassifier(hidden_dim=self.config.verifier_hidden_dim)
            default_path = DEFAULT_DEEPSET_VERIFIER_PATH
        else:
            raise ValueError("verifier_model_type must be one of: rule, RNN, DeepSet")

        model = model.to(self.config.device)
        checkpoint_path = self.config.verifier_model_path or default_path
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.config.device))
            if self.config.verbose:
                print(f"Loaded {self.config.verifier_model_type} verifier from {checkpoint_path}")
        else:
            print(f"Warning: verifier checkpoint not found at {checkpoint_path}. Using randomly initialized weights.")
        model.eval()
        return model

    def sample_score(self, sample: Dict[str, Any], verbose: bool = False) -> List[Dict[str, float]]:
        """Return T2B and B2A scores for every non-empty reasoning block."""
        token_to_block_scorer, tokenizer = make_token_to_block_scorer(self.infer_a)
        answer_text = sample["A"]
        scores: List[Dict[str, float]] = []

        for block_text in [block for block in sample["R"] if isinstance(block, str) and block.strip()]:
            token_ids = tokenizer(block_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
            if not token_ids:
                continue
            t2b_score = token_to_block_scorer(token_ids, block_text)
            b2a_score = self.infer_b.score_pair(block_text, answer_text)
            scores.append({"t2b": t2b_score, "b2a": b2a_score})
            if verbose:
                print(f"T2B={t2b_score:.4f} | B2A={b2a_score:.4f}")
        return scores

    def block_score(self, block_text: str, answer_text: str, verbose: bool = False) -> List[Dict[str, float]]:
        """Return T2B and B2A scores for a single block/answer pair."""
        token_to_block_scorer, tokenizer = make_token_to_block_scorer(self.infer_a)
        block_text = block_text.strip()
        token_ids = tokenizer(block_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        t2b_score = token_to_block_scorer(token_ids, block_text)
        b2a_score = self.infer_b.score_pair(block_text, answer_text)
        if verbose:
            print(f"T2B={t2b_score:.4f} | B2A={b2a_score:.4f}")
        return [{"t2b": t2b_score, "b2a": b2a_score}]

    def verify_samples(
        self,
        data: Iterable[Dict[str, Any]],
        probing_ratio: float,
        verification_threshold: float,
        *,
        token_ratio: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[int]:
        """Return binary verifier decisions for a batch of samples."""
        preprocessed = preprocess_samples_for_efficient_validation(
            data,
            self.infer_a,
            self.infer_b,
            token_ratio if token_ratio is not None else self.config.token_ratio,
            show_progress=show_progress,
        )

        decisions: List[int] = []
        if self.verifier_model is None:
            for sample in preprocessed:
                decisions.append(int(run_validation_on_preprocessed_rule_based(sample, probing_ratio, verification_threshold)))
        else:
            for sample in preprocessed:
                decisions.append(
                    int(
                        run_validation_on_preprocessed_learned(
                            sample,
                            probing_ratio,
                            verification_threshold,
                            self.verifier_model,
                            self.config.device,
                        )
                    )
                )
        return decisions

    def verifier_run(
        self,
        data: Iterable[Dict[str, Any]],
        token_ratio: float,
        probing_ratio: float,
        verification_threshold: float,
        verbose: bool = False,
    ) -> List[int]:
        """Backward-compatible wrapper around ``verify_samples``."""
        decisions = self.verify_samples(
            data,
            probing_ratio=probing_ratio,
            verification_threshold=verification_threshold,
            token_ratio=token_ratio,
            show_progress=verbose,
        )
        if verbose:
            print(f"Verification results: {decisions}")
        return decisions

    def verfier_run(self, *args, **kwargs) -> List[int]:
        """Deprecated spelling kept for compatibility with older scripts."""
        return self.verifier_run(*args, **kwargs)
