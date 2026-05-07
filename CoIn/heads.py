import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingHeadBase(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.proj = nn.Linear(embedding_dim * 4, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        h = torch.cat([a, b, a - b, a * b], dim=-1)
        out = self.proj(h)
        features["logits"] = self.classifier(out)
        return features

class MatchingHeadDeepMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        h = torch.cat([a, b, a - b, a * b], dim=-1)
        features["logits"] = self.mlp(h)
        return features

class MatchingHeadWithCosSim(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        cos_sim = F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)
        h = torch.cat([a, b, a - b, a * b, cos_sim], dim=-1)
        features["logits"] = self.fc(h)
        return features

class MatchingHeadResidual(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.proj = nn.Linear(embedding_dim * 4, hidden_dim)
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        h = torch.cat([a, b, a - b, a * b], dim=-1)
        x = self.proj(h)
        res = self.residual_block(x) + x
        features["logits"] = self.out(res)
        return features

class MatchingHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embedding_dim, 1)

    def forward(self, features):
        # Expecting B x T x D input tensors
        a, b = features["embedding_a"], features["embedding_b"]
        attn_output, _ = self.attn(a, b, b)  # Query=a, Key=b, Value=b
        pooled = attn_output.mean(dim=1)
        features["logits"] = self.proj(pooled)
        return features

class MatchingHeadWithCosSimV2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        input_dim = embedding_dim * 6 + 2 
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        cos_sim = F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)
        l2_dist = torch.norm(a - b, p=2, dim=-1).unsqueeze(-1)
        abs_diff = torch.abs(a - b)
        sq_diff = (a - b) ** 2

        h = torch.cat([
            a,
            b,
            a - b,
            a * b,
            abs_diff,
            sq_diff,
            cos_sim,
            l2_dist
        ], dim=-1)

        features["logits"] = self.fc(h)
        return features

class MatchingHeadWithCosSimDeeper(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        a, b = features["embedding_a"], features["embedding_b"]
        cos_sim = F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)
        h = torch.cat([a, b, a - b, a * b, cos_sim], dim=-1)
        features["logits"] = self.fc(h)
        return features


def get_matching_head(head_type, embedding_dim=256):
    if head_type == "base":
        return MatchingHeadBase(embedding_dim)
    elif head_type == "deep_mlp":
        return MatchingHeadDeepMLP(embedding_dim)
    elif head_type == "cos_sim":
        return MatchingHeadWithCosSim(embedding_dim)
    elif head_type == "residual":
        return MatchingHeadResidual(embedding_dim)
    elif head_type == "cross_attn":
        return MatchingHeadCrossAttention(embedding_dim)
    elif head_type == "feature":
        return MatchingHeadWithCosSimV2(embedding_dim)
    elif head_type == "cos_sim_deeper":
        return MatchingHeadWithCosSimDeeper(embedding_dim)

    else:
        raise ValueError(f"Unknown head type: {head_type}")
