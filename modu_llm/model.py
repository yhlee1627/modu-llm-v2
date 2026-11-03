"""8장: 빠른 GPT. 9장부터 학습과 생성에 이 모델을 씁니다.
기본 GPT와 계산 결과는 같고, 헤드를 한 번에 계산합니다."""
import torch
from torch import nn
from torch.nn import functional as F

from .model_basic import Embeddings, GPTConfig, TinyGPT, make_mlp


class CausalSelfAttention(nn.Module):
    """모든 헤드를 한 번에 계산하는 어텐션."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head                  # 헤드 하나의 크기
        qkv = self.qkv(x)                      # Q, K, V를 Linear 하나로 한꺼번에
        q, k, v = qkv.split(C, dim=2)
        # (B, T, C) → (B, 헤드 수, T, 헤드 크기): 헤드를 배치처럼 세우기
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # 점수 → 나누기 → 가리기 → 비율 → 모으기를 한 번에
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, 헤드 수, T, 헤드 크기) → (B, T, C): 헤드 결과 이어 붙이기
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class FastBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """빠른 GPT. 구조와 이름은 기본 GPT(TinyGPT)와 같습니다."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = Embeddings(cfg)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layer):
            self.blocks.append(FastBlock(cfg))
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(), targets.view(-1))
        return logits, loss


@torch.no_grad()
def copy_basic_to_fast(basic: TinyGPT, fast: GPT):
    """8.3절: 기본 GPT의 가중치를 빠른 GPT에 그대로 옮깁니다."""
    fast.emb.load_state_dict(basic.emb.state_dict())
    fast.ln.load_state_dict(basic.ln.state_dict())
    fast.head.load_state_dict(basic.head.state_dict())
    for b, f in zip(basic.blocks, fast.blocks):
        f.ln1.load_state_dict(b.ln1.state_dict())
        f.ln2.load_state_dict(b.ln2.state_dict())
        f.mlp.load_state_dict(b.mlp.state_dict())
        wq = torch.cat([h.query.weight for h in b.attn.heads], dim=0)
        wk = torch.cat([h.key.weight for h in b.attn.heads], dim=0)
        wv = torch.cat([h.value.weight for h in b.attn.heads], dim=0)
        f.attn.qkv.weight.copy_(torch.cat([wq, wk, wv], dim=0))
        f.attn.proj.load_state_dict(b.attn.proj.state_dict())
    return fast


def count_params(model):
    return sum(p.numel() for p in model.parameters())


__all__ = ["GPTConfig", "CausalSelfAttention", "FastBlock", "GPT",
           "copy_basic_to_fast", "count_params"]
