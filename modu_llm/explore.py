"""첫째·둘째 마당: 미리 학습시켜 둔 이야기 LLM을 불러와 들여다보는 도구.

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
from .real import _has_final, _split_repeat, _summary
from .settings import ASSET_REPO
from .sft import format_pair
from .vocab import Vocab

# 불러올 수 있는 모델: 이름 → 허깅페이스에 올려 둔 파일 이름
STAGES = {
    "before": None,                  # 학습 전(무작위 숫자로 채운 모델)
    "pretrain": "pretrain.pt",       # 이야기 데이터로 사전학습
    "sft": "sft.pt",                 # 요청을 따르도록 파인튜닝
    "feedback": "feedback.pt",       # 파인튜닝 모델을 선호 데이터 짝으로 한 번 더 학습(DPO)
    "step250": "ckpt_step250.pt",    # 사전학습 도중의 모습
    "step1000": "ckpt_step1000.pt",
    "step3000": "ckpt_step3000.pt",
}

STAGE_NAMES = {
    "before": "학습 전", "pretrain": "사전학습", "sft": "파인튜닝", "feedback": "피드백 학습",
    "step250": "사전학습 250번", "step1000": "사전학습 1,000번", "step3000": "사전학습 3,000번",
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
    def probs(self, text, temperature=1.0, top_k=None):
        """text 다음에 올 글자 모두(2,453자)의 확률.
        temperature로 점수를 나누고, top_k를 정하면 상위 k개 글자만 남겨 확률을 다시 나눕니다."""
        ids = self.vocab.encode(text) or [self.vocab.end_id]
        idx = torch.tensor([ids[-self.model.cfg.block_size:]])
        logits, _ = self.model(idx)
        last = logits[0, -1].float() / temperature
        if top_k is not None:
            kth = torch.topk(last, top_k).values[-1]
            last = last.masked_fill(last < kth, float("-inf"))
        return F.softmax(last, dim=-1)

    def top_chars(self, text, k=5, temperature=1.0, top_k=None):
        """확률이 높은 다음 글자 k개: [(글자, 확률), ...]"""
        p = self.probs(text, temperature, top_k)
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
    def _qk(self, text, layer, head):
        """layer층 head번 헤드의 질문 쪽지(q)와 명찰(k): 둘 다 (글자 수, 64)."""
        if not 1 <= layer <= len(self.model.blocks):
            raise ValueError(f"층은 1~{len(self.model.blocks)} 사이로 정하세요.")
        if not 1 <= head <= self.model.cfg.n_head:
            raise ValueError(f"헤드는 1~{self.model.cfg.n_head} 사이로 정하세요.")
        idx = torch.tensor([self.vocab.encode(text)])
        x = self.model.emb(idx)
        for block in self.model.blocks[:layer - 1]:
            x = block(x)
        block = self.model.blocks[layer - 1]
        h = block.ln1(x)
        B, T, C = h.shape
        a = block.attn
        hs = C // a.n_head
        q, k, _ = a.qkv(h).split(C, dim=2)
        q = q.view(B, T, a.n_head, hs)[0, :, head - 1]
        k = k.view(B, T, a.n_head, hs)[0, :, head - 1]
        return q, k

    def attention_weights(self, text, layer, head):
        """layer층 head번 헤드의 비율 표(글자 수 × 글자 수). layer, head는 1부터 셉니다."""
        q, k = self._qk(text, layer, head)
        T, hs = q.shape
        score = q @ k.T / math.sqrt(hs)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        score = score.masked_fill(~mask, float("-inf"))
        return F.softmax(score, dim=-1)

    @torch.no_grad()
    def layer_probs(self, text):
        """임베딩만 거친 뒤와 블록 1~6층을 지날 때마다, 그 자리에서 맞힌 다음 글자 확률 목록."""
        idx = torch.tensor([self.vocab.encode(text)[-self.model.cfg.block_size:]])
        x = self.model.emb(idx)
        out = []
        for i in range(len(self.model.blocks) + 1):
            if i > 0:
                x = self.model.blocks[i - 1](x)
            logits = self.model.head(self.model.ln(x))
            out.append(F.softmax(logits[0, -1].float(), dim=-1))
        return out

    def _chars(self, text):
        """사전에 있는 글자만 남긴 목록."""
        return [c for c in text if c in self.vocab.stoi]

    # ── 보여 주기 (화면에 출력) ─────────────────────────

    def next_chars(self, text, k=5, temperature=1.0, top_k=None):
        """text 다음 글자 후보 k개와 확률을 막대로 보여 줍니다.
        temperature(온도)와 top_k를 바꾸면 확률이 어떻게 달라지는지 볼 수 있습니다."""
        if top_k is not None:
            k = min(k, top_k)          # top-k 밖의 글자는 확률이 0이므로 보여 주지 않음
        rows = self.top_chars(text, k, temperature, top_k)
        extra = []
        if temperature != 1.0:
            extra.append(f"온도 {temperature}")
        if top_k is not None:
            extra.append(f"top-k {top_k}")
        tail = f" ({', '.join(extra)})" if extra else ""
        print(f"'{text}' 다음 글자 후보{tail}")
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

    def losses(self, text, repeat=1):
        """글자마다 손실 표와 문장 손실(평균)을 보여 줍니다.
        repeat=2: text를 띄어쓰기를 사이에 두고 두 번 넣고, 첫 번째와 두 번째의 손실을 나란히 보여 줍니다.
        (두 문구 모두 첫 글자는 빼고 나란히 봅니다.)"""
        if repeat == 1:
            rows = self.char_losses(text)
            print(f"{pad('글자')} {'확률':>5} {'손실':>5}")
            for ch, p, l in rows:
                print(f"{pad(show_char(ch))} {p:7.3f} {l:7.2f}")
            print(f"문장 손실(평균): {sum(r[2] for r in rows) / len(rows):.2f}")
            return
        if repeat != 2:
            raise ValueError("repeat는 1이나 2로 정하세요.")
        n = len(self._chars(text))
        if n < 2:
            raise ValueError("문구가 너무 짧습니다. 두 글자 이상인 문구로 해 보세요.")
        rows = self.char_losses(text + " " + text)      # rows[k]는 (k+1)번째 글자
        first, second = rows[0:n - 1], rows[n + 1:]
        josa = "을" if _has_final(text) else "를"
        print(f"'{text}'{josa} 두 번 넣고 글자마다 손실 비교")
        print("글자    첫 번째   두 번째")
        for a, b in zip(first, second):
            print(pad(show_char(a[0]), 6) + f"{a[2]:8.2f}{b[2]:10.2f}")
        ma = sum(a[2] for a in first) / len(first)
        mb = sum(b[2] for b in second) / len(second)
        print(pad("평균", 6) + f"{ma:8.2f}{mb:10.2f}")

    def attention(self, text, layer=1, head=1, digits=2, total=False):
        """layer층 head번 헤드의 비율 표를 색칠한 표로 보여 줍니다.
        줄 = 지금 글자, 세로줄 = 참고하는 글자. total=True면 줄마다 비율의 합을 덧붙입니다."""
        import pandas as pd
        w = self.attention_weights(text, layer, head)
        labels = [f"{show_char(c)}" for c in self._chars(text)]
        # 같은 글자가 여러 번 나와도 줄 이름이 겹치지 않게 자리 번호를 붙입니다.
        labels = [f"{i + 1}:{c}" for i, c in enumerate(labels)]
        df = pd.DataFrame(w.numpy().round(digits), index=labels, columns=labels)
        if total:
            df["합"] = w.sum(-1).numpy().round(digits)
        return df.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=1,
                                            subset=labels).format(f"{{:.{digits}f}}")

    def attention_steps(self, text, layer=1, head=1, at=None):
        """글자 하나(at번째, 1부터)가 앞 글자를 참고하는 계산을 단계마다 보여 줍니다.
        ① 점수 → ② 나누기 → ③ 가리기 → ④ 비율. at을 비우면 마지막 글자."""
        chars = self._chars(text)
        q, k = self._qk(text, layer, head)
        T, hs = q.shape
        t = T if at is None else at
        score = (k @ q[t - 1]).tolist()
        scaled = [x / math.sqrt(hs) for x in score]
        ratio = self.attention_weights(text, layer, head)[t - 1].tolist()
        print(f"{t}번째 글자 '{show_char(chars[t - 1])}'가 참고하는 계산 (층 {layer}, 헤드 {head})")
        print(f"{pad('참고하는 글자', 14)} {'①점수':>6} {'②나누기':>6} {'③가리기':>6} {'④비율':>5}")
        for j in range(T):
            name = pad(f"{j + 1}:{show_char(chars[j])}", 14)
            if j < t:
                print(f"{name} {score[j]:8.1f} {scaled[j]:8.2f} {scaled[j]:8.2f} {ratio[j]:7.2f}")
            else:
                print(f"{name} {score[j]:8.1f} {scaled[j]:8.2f} {'가림':>6} {ratio[j]:7.2f}")
        print(f"{pad('', 14)} {'':>8} {'':>8} {'합':>7} {sum(ratio):7.2f}")

    def looks_at(self, text, layer=1, head=1):
        """글자마다 가장 많이 참고하는 글자와 그 비율을 보여 줍니다."""
        chars = self._chars(text)
        w = self.attention_weights(text, layer, head)
        print(f"층 {layer} 헤드 {head}: 글자마다 가장 많이 참고하는 글자")
        for t in range(len(chars)):
            j = int(w[t].argmax())
            print(f"{pad(f'{t + 1}:{show_char(chars[t])}', 6)} → {pad(f'{j + 1}:{show_char(chars[j])}', 6)} {w[t, j].item():.2f}")

    def heads_summary(self, text, top=None, by=None):
        """모든 층·헤드가 무엇을 주로 참고하는지 평균 비율로 보여 줍니다.
        top과 by를 주면(14장) 진짜 LLM과 같은 표로, by 칸이 큰 헤드 top개만 보여 줍니다.
        맨 앞(첫 글자) · 자기 · 바로 앞 · (따라 하기) · 그 밖.
        top과 by를 비우면(5장) 아래 여섯 칸의 표를 모든 헤드에 대해 보여 줍니다.
        참고하는 글자를 다음 차례로 한 칸에만 넣으므로 한 줄의 합은 1입니다.
        자기 자신 → 바로 앞 글자 → 첫 글자 → 띄어쓰기 → 문장부호(.,!?) → 그 밖
        '가장 큰 것'은 '그 밖'을 뺀 다섯 칸 가운데 가장 큰 칸입니다."""
        if top is not None or by is not None:
            return self._heads_top(text, top or 5, by or "바로 앞")
        chars = self._chars(text)
        T = len(chars)
        names = ["자기", "바로 앞", "첫 글자", "띄어쓰기", "문장부호", "그 밖"]

        def kind(j, t):
            if j == t:
                return 0
            if j == t - 1:
                return 1
            if j == 0:
                return 2
            if chars[j] == " ":
                return 3
            if chars[j] in ".,!?":
                return 4
            return 5

        print(f"{pad('', 10)}" + "".join(f"{pad(n, 9)}" for n in names) + "가장 큰 것")
        for layer in range(1, len(self.model.blocks) + 1):
            for head in range(1, self.model.cfg.n_head + 1):
                w = self.attention_weights(text, layer, head)
                vals = [0.0] * 6
                for t in range(1, T):
                    for j in range(t + 1):
                        vals[kind(j, t)] += w[t, j].item() / (T - 1)
                best = names[max(range(5), key=lambda i: vals[i])]   # '그 밖'은 빼고 고름
                print(f"{pad(f'층{layer} 헤드{head}', 10)}" + "".join(f"{v:<9.2f}" for v in vals) + best)

    def _heads_top(self, text, top, by):
        """14장: 진짜 LLM의 heads_summary와 같은 표."""
        L, H = len(self.model.blocks), self.model.cfg.n_head
        atts = [torch.stack([self.attention_weights(text, l, h) for h in range(1, H + 1)])
                for l in range(1, L + 1)]
        phrase = _split_repeat(text)
        if phrase is not None:
            n = len(self._chars(phrase))
            if n < 3:
                raise ValueError("문구가 너무 짧습니다. 세 글자 이상인 문구로 해 보세요.")
            q0 = n + 1
            queries = [q0 + k for k in range(1, n - 1)]
            copy_of = {q0 + k: k + 1 for k in range(1, n - 1)}
        else:
            queries = list(range(2, len(self._chars(text))))
            copy_of = None
        _summary(atts, queries, copy_of, top, by)

    def layer_guess(self, text, k=3):
        """임베딩만 거쳤을 때와 블록을 한 층씩 지날 때마다 다음 글자를 맞혀 보게 합니다."""
        print(f"'{text}' 다음 글자, 층마다 맞혀 보기")
        for i, p in enumerate(self.layer_probs(text)):
            top = torch.topk(p, k)
            name = "임베딩만" if i == 0 else f"층 {i}"
            cells = [f"{pad(show_char(self.vocab.chars[j]))}{v * 100:3.0f}%"
                     for v, j in zip(top.values.tolist(), top.indices.tolist())]
            print(f"{pad(name, 9)} " + "   ".join(cells))

    def info(self):
        """모델의 크기와 파라미터 수를 보여 줍니다."""
        cfg = self.model.cfg
        n = lambda m: sum(p.numel() for p in m.parameters())
        total = n(self.model)
        print(f"{self}의 크기")
        print(f"글자 사전 {cfg.vocab_size:,}자 · 한 번에 읽는 글자 {cfg.block_size}자 · 숫자 카드 길이 {cfg.n_embd}")
        print(f"블록 {cfg.n_layer}층 · 블록마다 헤드 {cfg.n_head}개")
        print(f"파라미터 {total:,}개")
        print(f"  숫자 카드(글자 카드와 자리 카드)  {n(self.model.emb):>9,}")
        for i, b in enumerate(self.model.blocks):
            print(f"  블록 {i + 1}층{'':24}{n(b):>9,}")
        print(f"  출력층(마지막 정리와 글자 점수){'':2}{n(self.model.ln) + n(self.model.head):>9,}")

    def compare_losses(self, texts):
        """문장마다 손실(모델이 놀란 정도)을 보여 줍니다."""
        print("손실   문장")
        for t in texts:
            print(f"{self.loss(t):5.2f}  {t}")

    # 3장: 글자 → 번호 → 숫자 카드

    def to_ids(self, text):
        """글자마다 글자 사전의 번호를 보여 줍니다."""
        chars = self._chars(text)
        ids = [self.vocab.stoi[c] for c in chars]
        w = max(4, max(len(str(i)) for i in ids))
        print("글자: " + " ".join(pad(show_char(c), w) for c in chars))
        print("번호: " + " ".join(f"{i:<{w}}" for i in ids))
        missing = self.vocab.unknown_chars(text)
        if missing:
            print("사전에 없어 건너뛴 글자: " + " ".join(missing))

    def to_text(self, ids):
        """번호를 글자로 바꿔 보여 줍니다."""
        print(self.vocab.decode(ids))

    @torch.no_grad()
    def cards(self, text, n=3):
        """글자마다 글자 카드, 자리 카드, 둘을 더한 카드의 앞 n개 숫자를 보여 줍니다."""
        chars = self._chars(text)
        ids = torch.tensor([self.vocab.stoi[c] for c in chars])
        tok = self.model.emb.token_emb(ids)
        pos = self.model.emb.pos_emb(torch.arange(len(chars)))
        fmt = lambda v: "[" + " ".join(f"{x:+.2f}" for x in v[:n].tolist()) + "]"
        print(f"카드 하나에 숫자 {tok.shape[1]}개. 앞의 {n}개만 보입니다.")
        print(f"{pad('자리·글자', 10)}{pad('글자 카드', 20)}  {pad('자리 카드', 20)}  합친 카드")
        for i, c in enumerate(chars):
            print(f"{pad(f'{i + 1}:{show_char(c)}', 10)}{fmt(tok[i])} + {fmt(pos[i])} = {fmt(tok[i] + pos[i])}")

    def compare_tokens(self, text, *names):
        """이야기 LLM과 다른 LLM들(허깅페이스 이름, 여러 개 가능)이 같은 글을 토큰으로 나누는 모습을 비교합니다."""
        from transformers import AutoTokenizer
        mine = [show_char(c) for c in self._chars(text)]
        print(f"이야기 LLM: 토큰 {len(mine)}개 (사전 {len(self.vocab):,}개)")
        print("|".join(mine))
        missing = self.vocab.unknown_chars(text)
        if missing:
            print("사전에 없어 건너뛴 글자: " + " ".join(missing))
        for name in names:
            # clean_up_tokenization_spaces=False: 일부 토크나이저가 내는 경고를 없앱니다(출력은 같음).
            tok = AutoTokenizer.from_pretrained(name, clean_up_tokenization_spaces=False)
            ids = tok.encode(text, add_special_tokens=False)
            pieces = [tok.decode([i]).replace(" ", "␣") for i in ids]
            print(f"{name.split('/')[-1]}: 토큰 {len(ids)}개 (사전 {len(tok):,}개)")
            print("|".join(pieces))




def load_story_llm(stage="pretrain", seed=42, verbose=True):
    """미리 학습시켜 둔 이야기 LLM을 불러옵니다.
    stage: "pretrain"(사전학습), "sft"(파인튜닝), "feedback"(피드백 학습), "before"(학습 전),
           "step250"·"step1000"·"step3000"(사전학습 도중)
    seed: 글을 뽑을 때 쓰는 난수의 시작값. 같은 seed로 처음부터 실행하면 같은 글이 나옵니다.
    verbose: False면 "준비 완료"를 출력하지 않습니다."""
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
    if verbose:
        print(f"{llm} 준비 완료")
    return llm
