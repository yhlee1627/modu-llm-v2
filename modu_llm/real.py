"""넷째 마당(13·14장): 허깅페이스의 진짜 LLM을 불러와 이야기 LLM과 같은 방법으로 들여다보는 도구.

    llm = load_llm("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    llm.next_chars("대한민국의 수도는")         # 다음 토큰 후보와 확률
    llm.layer_guess("대한민국의 수도는")        # 층마다 다음 토큰 맞혀 보기

이야기 LLM과 같은 이름의 함수를 씁니다. 다른 점은 글자 대신 토큰을 다룬다는 것입니다.
- 글 맨 앞에 '시작' 토큰을 하나 붙여서 넣습니다(모델이 배울 때와 같은 모양).
- 층과 헤드는 1부터 셉니다. 토큰 자리도 1부터 세고, 시작 토큰은 0번 자리입니다.
- CPU에서 float32로 계산합니다. 같은 seed로 처음부터 실행하면 같은 글이 나옵니다.
"""
import math
import unicodedata

import torch
from torch.nn import functional as F


def _width(s):
    """화면에서 차지하는 칸 수(한글·한자는 2칸)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, n):
    return s + " " * max(0, n - _width(s))


def _has_final(s, default=True):
    """마지막 한글 글자에 받침이 있는지(을/를, 이/가 고르기)."""
    for c in reversed(s):
        if "가" <= c <= "힣":
            return (ord(c) - 0xAC00) % 28 != 0
    return default


def _split_repeat(text):
    """text가 '문구 + 띄어쓰기 + 같은 문구'이면 그 문구를, 아니면 None을 돌려줍니다."""
    m = len(text) // 2
    if len(text) % 2 == 1 and text[m] == " " and text[:m] == text[m + 1:] and m > 0:
        return text[:m]
    return None


class RealLLM:
    """허깅페이스의 진짜 LLM 하나(모델 + 토크나이저)."""

    def __init__(self, name, model, tokenizer):
        self.name = name
        self.short = name.split("/")[-1]
        self.model = model.eval()
        self.tok = tokenizer
        dec = model.model
        if hasattr(dec, "language_model"):
            dec = dec.language_model
        self.layers = dec.layers
        self.norm = dec.norm
        self.embed = dec.embed_tokens
        self.head = model.get_output_embeddings()
        self.cfg = getattr(model.config, "text_config", model.config)
        tk = tokenizer
        self.start = tk.bos_token_id if tk.bos_token_id is not None else tk.convert_tokens_to_ids("<|endoftext|>")
        self.stop = {self.start}
        ends = [tk.eos_token_id, getattr(getattr(model, "generation_config", None), "eos_token_id", None)]
        for e in ends:
            for i in (e if isinstance(e, (list, tuple)) else [e]):
                if i is not None:
                    self.stop.add(int(i))

    def __repr__(self):
        return f"진짜 LLM({self.short})"

    # ── 계산 ─────────────────────────────────────────

    def piece(self, i):
        """토큰 번호 하나를 눈에 보이는 조각으로(띄어쓰기는 ␣, 줄바꿈은 ↵)."""
        return self.tok.decode([i]).replace(" ", "␣").replace("\n", "↵")

    def token_ids(self, text):
        """text의 토큰 번호(시작 토큰은 빼고)."""
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def ids(self, text):
        """맨 앞에 시작 토큰을 붙인 토큰 번호."""
        return [self.start] + self.token_ids(text)

    @torch.no_grad()
    def _run(self, ids, **kw):
        return self.model(torch.tensor([ids]), **kw)

    def probs(self, text, temperature=1.0, top_k=None):
        """text 다음에 올 토큰 모두의 확률."""
        last = self._run(self.ids(text)).logits[0, -1].float() / temperature
        if top_k is not None:
            kth = torch.topk(last, top_k).values[-1]
            last = last.masked_fill(last < kth, float("-inf"))
        return F.softmax(last, dim=-1)

    def token_losses(self, ids):
        """[시작]+토큰 번호를 넣고 토큰마다 (조각, 확률, 손실). 첫 토큰은 시작 토큰을 보고 맞힌 것."""
        logits = self._run(ids).logits[0, :-1].float()
        lp = F.log_softmax(logits, dim=-1)
        rows = []
        for t, target in enumerate(ids[1:]):
            loss = -lp[t, target].item()
            rows.append((self.piece(target), math.exp(-loss), loss))
        return rows

    def loss(self, text):
        """문장 손실: 토큰마다 손실의 평균."""
        rows = self.token_losses(self.ids(text))
        return sum(r[2] for r in rows) / len(rows)

    def attentions(self, ids):
        """층마다 비율 표: [층](헤드, 자리, 자리). 자리 0은 시작 토큰."""
        out = self._run(ids, output_attentions=True)
        return [a[0].float() for a in out.attentions]

    def _repeat_tokens(self, phrase, need=2):
        """phrase를 띄어쓰기를 사이에 두고 두 번 넣었을 때 토큰이 같게 나뉘는지 확인하고 phrase의 토큰 수를 돌려줍니다."""
        one = self.token_ids(phrase)
        full = self.token_ids(phrase + " " + phrase)
        n = len(one)
        if n < need:
            raise ValueError(f"문구가 너무 짧습니다. 토큰 {need}개 이상인 문구로 해 보세요(지금 {n}개).")
        if full[:n] != one or full[n + 1:] != one[1:]:
            raise ValueError("두 번 넣었을 때 토큰이 같게 나뉘지 않는 문구입니다. 다른 문구로 해 보세요.")
        return n

    def _check(self, layer, head):
        L, H = len(self.layers), self.cfg.num_attention_heads
        if not 1 <= layer <= L:
            raise ValueError(f"층은 1~{L} 사이로 정하세요.")
        if not 1 <= head <= H:
            raise ValueError(f"헤드는 1~{H} 사이로 정하세요.")

    @torch.no_grad()
    def layer_probs(self, text):
        """블록을 한 층씩 지날 때마다, 그 자리에서 맞힌 다음 토큰 확률 목록(층 1~L)."""
        outs = {}
        hooks = [layer.register_forward_hook(
            lambda m, a, o, i=i: outs.__setitem__(i, o[0] if isinstance(o, tuple) else o))
            for i, layer in enumerate(self.layers)]
        try:
            self._run(self.ids(text))
        finally:
            for h in hooks:
                h.remove()
        cap = getattr(self.cfg, "final_logit_softcapping", None)
        result = []
        for i in range(len(self.layers)):
            logits = self.head(self.norm(outs[i][0, -1])).float()
            if cap:
                logits = torch.tanh(logits / cap) * cap
            result.append(F.softmax(logits, dim=-1))
        return result

    # ── 보여 주기 ─────────────────────────────────────

    def info(self):
        """모델의 크기와 파라미터 수를 보여 줍니다."""
        n = lambda m: sum(p.numel() for p in m.parameters())
        total = n(self.model)
        emb = self.embed.weight.numel()
        per_layer = n(self.layers[0])
        L = len(self.layers)
        tied = self.head.weight.data_ptr() == self.embed.weight.data_ptr()
        out_layer = n(self.norm) + (0 if tied else self.head.weight.numel())
        print(f"{self.short}의 크기")
        print(f"토큰 사전 {len(self.tok):,}개 · 한 번에 읽는 토큰 {self.cfg.max_position_embeddings:,}개 · "
              f"숫자 카드 길이 {self.cfg.hidden_size:,}")
        print(f"블록 {L}층 · 블록마다 헤드 {self.cfg.num_attention_heads}개")
        print(f"파라미터 {total:,}개")
        print(_pad("  숫자 카드(토큰 카드)", 37) + f"{emb:>11,}")
        print(_pad(f"  블록 {L}층(한 층에 {per_layer:,})", 37) + f"{per_layer * L:>11,}")
        print(_pad("  출력층(마지막 정리)", 37) + f"{out_layer:>11,}")

    def next_chars(self, text, k=5, temperature=1.0, top_k=None):
        """text 다음 토큰 후보 k개와 확률을 막대로 보여 줍니다."""
        if top_k is not None:
            k = min(k, top_k)
        p = self.probs(text, temperature, top_k)
        top = torch.topk(p, k)
        rows = [(self.piece(i), v) for v, i in zip(top.values.tolist(), top.indices.tolist())]
        width = max([5] + [_width(s) for s, _ in rows])
        print(f"'{text}' 다음 토큰 후보")
        for n, (s, v) in enumerate(rows, 1):
            print(f"{n:>2}. {_pad(s, width)} {v * 100:5.1f}%  {'█' * max(1, round(v * 50))}")
        rest = 1 - sum(v for _, v in rows)
        print(f"    나머지 {len(self.tok) - k:,}토큰 {rest * 100:4.1f}%")

    @torch.no_grad()
    def write(self, text, length=100, temperature=0.8, top_k=None):
        """text에 이어서 토큰 length개까지 씁니다(끝 토큰이 나오면 멈춤)."""
        ids = self.ids(text)
        new = []
        past, x = None, torch.tensor([ids])
        for _ in range(length):
            out = self.model(x, past_key_values=past, use_cache=True)   # 앞에서 계산한 것은 다시 쓰고 새 토큰만 계산
            past = out.past_key_values
            last = out.logits[0, -1].float() / temperature
            if top_k is not None:
                kth = torch.topk(last, top_k).values[-1]
                last = last.masked_fill(last < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(last, dim=-1), 1).item()
            if nxt in self.stop:
                break
            new.append(nxt)
            x = torch.tensor([[nxt]])
        print(text + self.tok.decode(new))

    def losses(self, text, repeat=1):
        """토큰마다 손실 표와 문장 손실(평균)을 보여 줍니다.
        repeat=2: text를 띄어쓰기를 사이에 두고 두 번 넣고, 첫 번째와 두 번째의 손실을 나란히 보여 줍니다.
        (두 문구 모두 첫 토큰은 빼고 나란히 봅니다.)"""
        if repeat == 1:
            rows = self.token_losses(self.ids(text))
            print(f"{_pad('토큰', 4)} {'확률':>5} {'손실':>5}")
            for s, p, l in rows:
                print(f"{_pad(s, 4)} {p:7.3f} {l:7.2f}")
            print(f"문장 손실(평균): {sum(r[2] for r in rows) / len(rows):.2f}")
            return
        if repeat != 2:
            raise ValueError("repeat는 1이나 2로 정하세요.")
        n = self._repeat_tokens(text)
        full = self.token_ids(text + " " + text)
        r = self.token_losses([self.start] + full)
        first, second = r[1:n], r[n + 1:]
        josa = "을" if _has_final(text) else "를"
        print(f"'{text}'{josa} 두 번 넣고 토큰마다 손실 비교")
        print("토큰    첫 번째   두 번째")
        for a, b in zip(first, second):
            print(_pad(a[0], 6) + f"{a[2]:8.2f}{b[2]:10.2f}")
        ma = sum(a[2] for a in first) / len(first)
        mb = sum(b[2] for b in second) / len(second)
        print(_pad("평균", 6) + f"{ma:8.2f}{mb:10.2f}")

    def compare_losses(self, texts):
        """문장마다 손실(모델이 놀란 정도)을 보여 줍니다."""
        print("손실   문장")
        for t in texts:
            print(f"{self.loss(t):5.2f}  {t}")

    def attention(self, text, layer=1, head=1, digits=2):
        """layer층 head번 헤드의 비율 표를 색칠한 표로 보여 줍니다.
        줄 = 지금 토큰, 세로줄 = 참고하는 토큰. 시작 토큰의 줄과 세로줄은 빼고 보여 주므로
        줄의 합이 1보다 작을 수 있습니다(나머지는 시작 토큰을 본 비율)."""
        import pandas as pd
        self._check(layer, head)
        ids = self.ids(text)
        w = self.attentions(ids)[layer - 1][head - 1][1:, 1:]
        labels = [f"{i}:{self.piece(t)}" for i, t in enumerate(ids[1:], 1)]
        df = pd.DataFrame(w.numpy().round(digits), index=labels, columns=labels)
        return df.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=1).format(f"{{:.{digits}f}}")

    def looks_at(self, text, layer=1, head=1):
        """토큰마다 가장 많이 참고하는 토큰과 그 비율을 보여 줍니다(시작 토큰은 0:시작)."""
        self._check(layer, head)
        ids = self.ids(text)
        w = self.attentions(ids)[layer - 1][head - 1]
        print(f"층 {layer} 헤드 {head}: 토큰마다 가장 많이 참고하는 토큰")
        for i in range(1, len(ids)):
            j = int(w[i].argmax())
            target = "0:시작" if j == 0 else f"{j}:{self.piece(ids[j])}"
            print(_pad(f"{i}:{self.piece(ids[i])}", 9) + "→ " + _pad(target, 9) + f"{w[i, j].item():.2f}")

    def heads_summary(self, text, top=5, by="바로 앞"):
        """모든 층·헤드의 평균 비율을 재고, by 칸이 큰 헤드 top개를 보여 줍니다.
        맨 앞(시작 토큰) · 자기 · 바로 앞 · 그 밖. text가 '문구 + 띄어쓰기 + 같은 문구'이면
        '따라 하기'(두 번째 문구의 토큰이 첫 번째 문구에서 같은 토큰의 바로 다음 토큰을 보는 비율) 칸이 생깁니다."""
        ids = self.ids(text)
        phrase = _split_repeat(text)
        if phrase is not None:
            n = self._repeat_tokens(phrase, need=3)
            q0 = 1 + n
            queries = [q0 + k for k in range(1, n - 1)]
            copy_of = {q0 + k: 1 + k + 1 for k in range(1, n - 1)}
        else:
            queries = list(range(2, len(ids)))
            copy_of = None
        atts = self.attentions(ids)
        _summary(atts, queries, copy_of, top, by)

    def layer_guess(self, text, k=3):
        """블록을 한 층씩 지날 때마다 다음 토큰을 맞혀 보게 합니다.
        층마다 그 자리의 결과를 마지막 정리(norm)와 출력층에 바로 넣어 봅니다."""
        print(f"'{text}' 다음 토큰, 층마다 맞혀 보기")
        for j, p in enumerate(self.layer_probs(text), 1):
            top = torch.topk(p, k)
            cells = [_pad(f"{self.piece(i)} {v * 100:.1f}%", 18) for v, i in zip(top.values.tolist(), top.indices.tolist())]
            print((_pad(f"층 {j}", 7) + "".join(cells)).rstrip())


def _summary(atts, queries, copy_of, top, by, label_width=None):
    """heads_summary 공통 부분. atts: [층](헤드, 자리, 자리), 0번 자리가 '맨 앞'."""
    names = ["맨 앞", "자기", "바로 앞"] + (["따라 하기"] if copy_of else []) + ["그 밖"]
    if by not in names[:-1]:
        if by == "따라 하기":
            raise ValueError("'따라 하기'는 같은 문구를 띄어쓰기를 사이에 두고 두 번 넣었을 때만 잽니다.")
        raise ValueError(f"by는 {', '.join(names[:-1])} 가운데 하나로 정하세요.")
    if not queries:
        raise ValueError("글이 너무 짧아 잴 자리가 없습니다. 더 긴 글로 해 보세요.")
    L, H = len(atts), atts[0].shape[0]
    rows = []
    for l in range(L):
        for h in range(H):
            a = atts[l][h]
            vals = [0.0] * len(names)
            for q in queries:
                parts = [a[q, 0].item(), a[q, q].item(), a[q, q - 1].item()]
                if copy_of:
                    parts.append(a[q, copy_of[q]].item())
                parts.append(1 - sum(parts))
                for i, v in enumerate(parts):
                    vals[i] += v / len(queries)
            rows.append((f"층{l + 1} 헤드{h + 1}", vals))
    key = names.index(by)
    rows.sort(key=lambda r: -r[1][key])
    W = label_width or max(10, _width(f"층{L} 헤드{H}"))
    widths = [7, 7, 8, 10, 11] if copy_of else [7, 7, 8, 9]
    josa = "이" if _has_final(by) else "가"
    print(f"헤드 {L * H}개 가운데 '{by}'{josa} 큰 헤드 {top}개")
    print(" " * (W + 2) + "   ".join(names))
    for label, vals in rows[:top]:
        print(_pad(label, W) + "".join(f"{v:{w}.2f}" for v, w in zip(vals, widths)))





def load_llm(name, seed=42, verbose=True):
    """허깅페이스의 LLM을 이름으로 불러옵니다.
    예: load_llm("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    처음에는 모델 파일을 내려받느라 시간이 걸립니다(0.5B 모델 약 1.1GB).
    약관 동의가 필요한 모델(예: google/gemma-3-1b-pt)은 허깅페이스에서 약관에 동의하고,
    코랩에서 notebook_login()으로 로그인(주소를 열고 8자리 코드 입력)한 뒤 불러옵니다(부록 C)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()          # 불러올 때 나오는 안내문은 숨깁니다.
    hf_logging.disable_progress_bar()
    try:
        tok = AutoTokenizer.from_pretrained(name, clean_up_tokenization_spaces=False)
        kw = dict(attn_implementation="eager")   # 비율 표를 꺼내려면 eager 방식이 필요합니다.
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, **kw)
    finally:
        hf_logging.enable_progress_bar()
        hf_logging.set_verbosity(level)
    torch.manual_seed(seed)
    llm = RealLLM(name, model, tok)
    if verbose:
        print(f"{llm.short} 준비 완료")
    return llm
