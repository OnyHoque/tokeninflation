import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from sentence_transformers import SentenceTransformer
from heads import get_matching_head

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MatchingInference:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        print(f"Loading embedding model from {model_dir}/embedding_model ...")
        self.embedding_model = SentenceTransformer(f"{model_dir}/embedding_model", trust_remote_code=True, device=DEVICE)
        self.embedding_model.eval()
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.matching_head = get_matching_head("cos_sim", embedding_dim)
        head_path = os.path.join(model_dir, "matching_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"matching_head.pt not found in {model_dir}")
        self.matching_head.load_state_dict(torch.load(head_path, map_location=DEVICE))
        self.matching_head = self.matching_head.to(DEVICE).eval()
        self.tokenid2emb = self._build_token_embedding_cache(model_dir)

    def _build_token_embedding_cache(self, model_dir: str) -> dict:
        cache_path = os.path.join(model_dir, "tokenid_embedding_cache.pt")
        if os.path.exists(cache_path):
            print(f"📦 Loading token ID embedding cache from {cache_path} ...")
            tokenid2emb_raw = torch.load(cache_path, map_location=DEVICE)
            # Ensure all tensors are on DEVICE and float32
            tokenid2emb = {int(k): v.to(DEVICE).to(torch.float32) for k, v in tokenid2emb_raw.items()}
            return tokenid2emb

        # Otherwise, build from scratch (this can be slow!)
        tokenizer = self.embedding_model.tokenizer
        vocab = tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}

        # Filter to a reasonable ID range if you want, or keep all
        token_ids = sorted(inv_vocab.keys())
        print(f"Found {len(token_ids)} tokens in tokenizer vocab.")
        tokens = [inv_vocab[i] for i in token_ids]
        filtered = [(tid, tok) for tid, tok in zip(token_ids, tokens)]
        print(f"Encoding {len(filtered)} tokens to build cache...")
        tokens = [x[1] for x in filtered]
        print(f"Encoding {len(tokens)} tokens to build cache...")
        token_embs = self.embedding_model.encode(tokens, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
        tokenid2emb = {tid: token_embs[i].detach().clone().to(DEVICE).to(torch.float32) for i, (tid, _) in enumerate(filtered)}
        torch.save(tokenid2emb, cache_path)
        print(f"Saved token ID embedding cache to {cache_path}")
        return tokenid2emb
    
    def embedding(self, text: str) -> torch.Tensor:
        return autograd_safe(
            self.embedding_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=DEVICE,
            )
        )


    def matching_score(self, text_a: str, text_b: str) -> float:
        ea = self.embedding_model.encode(text_a, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE).unsqueeze(0)
        eb = self.embedding_model.encode(text_b, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE).unsqueeze(0)
        out = self.matching_head({"embedding_a": ea, "embedding_b": eb})
        return float(torch.sigmoid(out["logits"]).item())
    
def autograd_safe(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().to(DEVICE).to(torch.float32)
