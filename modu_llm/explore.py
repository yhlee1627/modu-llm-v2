"""첫째·둘째 마당: 저자가 학습시킨 이야기 LLM을 불러와 들여다보는 도구.

책의 1~7장 실습에서 씁니다. 코드는 한두 줄이고, 독자는 따옴표 안의 문장과 숫자만 바꿉니다.
    llm = load_story_llm()            # 사전학습을 마친 이야기 LLM
    llm.next_chars("어느 날, ")        # 다음 글자 후보와 확률
    llm.write("어느 날, ")             # 이어서 이야기 쓰기
"""
import math

import torch
from torch.nn import functional as F

from .generate import generate
from .model import GPT
from .model_basic import GPTConfig
from .settings import ASSET_REPO
from .sft import format_pair
from .vocab import Vocab

# 불러올 수 있는 모델: 이름 → 저자 허깅페이스의 파일 이름
STAGES = {
    "before": None,                  # 학습 전(무작위 숫자로 채운 모델)
    "pretrain": "pretrain.pt",       # 이야기 데이터로 사전학습
    "sft": "sft.pt",                 # 요청을 따르도록 파인튜닝
    "step250": "ckpt_step250.pt",    # 사전학습 도중의 모습
    "step1000": "ckpt_step1000.pt",
    "step3000": "ckpt_step3000.pt",
}

STAGE_NAMES = {
    "before": "학습 전", "pretrain": "사전학습", "sft": "파인튜닝",
    "step250": "사전학습 250걸음", "step1000": "사전학습 1,000걸음", "step3000": "사전학습 3,000걸음",
}


def show_char(ch):
    """눈에 보이지 않는 글자를 보이게 바꿉니다."""
    return {" ": "␣", "\n": "↵", "§": "§(끝)"}.get(ch, ch)


def pad(s, width=4):
    """한글처럼 두 칸을 차지하는 글자를 셈해 오른쪽을 빈칸으로 채웁니다."""
    w = sum(2 if "ᄀ" <= c <= "ￜ" and c not in "␣↵…‘’“”·" else 1 for c in s)
    return s + " " * max(0, width - w)


class StoryLLM:
    """이야기 LLM 하나(모델 + 글자 사전)."""

    def __init__(self, model, vocab, stage):
        self.model = model.eval()
        self.vocab = vocab
        self.stage = stage

    def __repr__(self):
        return f"이야기 LLM({STAGE_NAMES.get(self.stage, self.stage)})"

    # ── 계산 (값을 돌려줌) ─────────────────────────────

    @torch.no_grad()
    def probs(self, text):
        """text 다음에 올 글자 모두(2,453자)의 확률."""
        ids = self.vocab.encode(text) or [self.vocab.end_id]
        idx = torch.tensor([ids[-self.model.cfg.block_size:]])
        logits, _ = self.model(idx)
        return F.softmax(logits[0, -1].float(), dim=-1)

    def top_chars(self, text, k=5):
        """확률이 높은 다음 글자 k개: [(글자, 확률), ...]"""
        p = self.probs(text)
        top = torch.topk(p, k)
        return [(self.vocab.chars[i], v) for v, i in zip(top.values.tolist(), top.indices.tolist())]

    def continue_text(self, text, length=100, temperature=0.8, top_k=None):
        """text 뒤에 이어질 글(text는 빼고)."""
        return generate(self.model, self.vocab, text, max_new_tokens=length,
                        temperature=temperature, top_k=top_k)

    @torch.no_grad()
    def char_losses(self, text):
        """글자마다 손실: [(글자, 그 글자에 준 확률, 손실), ...]. 첫 글자는 앞 글자가 없어 빠집니다."""
        ids = self.vocab.encode(text)
        idx = torch.tensor([ids[:-1]])
        logits, _ = self.model(idx)
        p = F.softmax(logits[0].float(), dim=-1)
        rows = []
        for t, target in enumerate(ids[1:]):
            pt = p[t, target].item()
            rows.append((self.vocab.chars[target], pt, -math.log(pt)))
        return rows

    def loss(self, text):
        """문장 손실: 글자마다 손실의 평균."""
        rows = self.char_losses(text)
        return sum(r[2] for r in rows) / len(rows)

    @torch.no_grad()
    def attention_weights(self, text, layer, head):
        """layer층 head번 헤드의 비율 표(글자 수 × 글자 수). layer, head는 1부터 셉니다."""
        idx = torch.tensor([self.vocab.encode(text)])
        x = self.model.emb(idx)
        for i, block in enumerate(self.model.blocks):
            if i == layer - 1:
                h = block.ln1(x)
                B, T, C = h.shape
                a = block.attn
                hs = C // a.n_head
                q, k, _ = a.qkv(h).split(C, dim=2)
                q = q.view(B, T, a.n_head, hs)[0, :, head - 1]
                k = k.view(B, T, a.n_head, hs)[0, :, head - 1]
                score = q @ k.T / math.sqrt(hs)
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
                score = score.masked_fill(~mask, float("-inf"))
                return F.softmax(score, dim=-1)
            x = block(x)
        raise ValueError(f"층은 1~{len(self.model.blocks)} 사이로 정하세요.")

    # ── 보여 주기 (화면에 출력) ─────────────────────────

    def next_chars(self, text, k=5):
        """text 다음 글자 후보 k개와 확률을 막대로 보여 줍니다."""
        rows = self.top_chars(text, k)
        print(f"'{text}' 다음 글자 후보")
        for n, (ch, p) in enumerate(rows, 1):
            bar = "█" * round(p * 50)          # 막대 한 칸 = 2%
            print(f"{n:>2}. {pad(show_char(ch))} {p * 100:5.1f}%  {bar}")
        rest = 1 - sum(p for _, p in rows)
        print(f"    나머지 {len(self.vocab) - k:,}글자 {rest * 100:5.1f}%")

    def write(self, text, length=100, temperature=0.8, top_k=None, n=1):
        """text에 이어서 글을 씁니다. n번 쓰면 번호를 붙여 보여 줍니다."""
        for i in range(n):
            new = self.continue_text(text, length, temperature, top_k)
            head = f"[{i + 1}] " if n > 1 else ""
            print(head + text + new)
            if n > 1 and i < n - 1:
                print()

    def ask(self, request, length=200, temperature=0.8, top_k=None):
        """요청을 '요청: …/이야기: ' 형식에 넣어 이야기를 받습니다."""
        prompt = format_pair(request)
        print(prompt + self.continue_text(prompt, length, temperature, top_k))

    def losses(self, text):
        """글자마다 손실 표와 문장 손실(평균)을 보여 줍니다."""
        rows = self.char_losses(text)
        print(f"{pad('글자')} {'확률':>5} {'손실':>5}")
        for ch, p, l in rows:
            print(f"{pad(show_char(ch))} {p:7.3f} {l:7.2f}")
        print(f"문장 손실(평균): {sum(r[2] for r in rows) / len(rows):.2f}")

    def attention(self, text, layer=1, head=1, digits=2):
        """layer층 head번 헤드의 비율 표를 색칠한 표로 보여 줍니다.
        줄 = 지금 글자, 세로줄 = 참고하는 글자."""
        import pandas as pd
        w = self.attention_weights(text, layer, head)
        labels = [f"{show_char(c)}" for c in text]
        # 같은 글자가 여러 번 나와도 줄 이름이 겹치지 않게 자리 번호를 붙입니다.
        labels = [f"{i + 1}:{c}" for i, c in enumerate(labels)]
        df = pd.DataFrame(w.numpy().round(digits), index=labels, columns=labels)
        return df.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=1).format(f"{{:.{digits}f}}")


def load_story_llm(stage="pretrain", seed=42):
    """저자가 학습시킨 이야기 LLM을 불러옵니다.
    stage: "pretrain"(사전학습), "sft"(파인튜닝), "before"(학습 전),
           "step250"·"step1000"·"step3000"(사전학습 도중)
    seed: 글을 뽑을 때 쓰는 난수의 시작값. 같은 seed로 처음부터 실행하면 같은 글이 나옵니다."""
    if stage not in STAGES:
        raise ValueError(f"stage는 {list(STAGES)} 중 하나로 정하세요.")
    vocab = Vocab.build()
    if STAGES[stage] is None:
        torch.manual_seed(0)
        model = GPT(GPTConfig())
    else:
        from huggingface_hub import hf_hub_download
        from .train import load_checkpoint
        path = hf_hub_download(ASSET_REPO, STAGES[stage])
        model, _, _, _ = load_checkpoint(path, device="cpu")
    torch.manual_seed(seed)
    llm = StoryLLM(model, vocab, stage)
    print(f"{llm} 준비 완료")
    return llm
