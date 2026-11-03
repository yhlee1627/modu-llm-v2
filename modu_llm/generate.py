"""10장: 글 생성."""
import torch
from torch.nn import functional as F


def generate(model, vocab, prompt, max_new_tokens=300, temperature=0.8, top_k=None, stop_at_end=True):
    """prompt 뒤에 이어질 글을 만들어 돌려줍니다(prompt는 빼고).
    temperature: 낮을수록 확률 높은 글자에 집중, 높을수록 다양하게
    top_k: 정하면 확률 상위 k개 글자에서만 고름
    stop_at_end: 끝 표시(§)가 나오면 멈춤"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = vocab.encode(prompt)
    if len(ids) == 0:
        ids = [vocab.end_id]          # 빈 글이면 끝 표시(새 이야기의 시작)로 시작
    idx = torch.tensor([ids], device=device)
    new_ids = []
    for i in range(max_new_tokens):
        context = idx[:, -model.cfg.block_size:]      # 마지막 384글자만
        with torch.no_grad():
            logits, _ = model(context)
        last = logits[0, -1].float()                  # 마지막 자리의 점수
        last = last / temperature                     # ① 온도로 나누기
        if top_k is not None:                         # ② 상위 k개만 남기기
            kth = torch.topk(last, top_k).values[-1]
            last = last.masked_fill(last < kth, float("-inf"))
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, 1)         # 확률대로 하나 뽑기
        if stop_at_end and next_id.item() == vocab.end_id:
            break                                     # ③ 끝 표시가 나오면 멈추기
        new_ids.append(next_id.item())
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
    model.train(was_training)
    return vocab.decode(new_ids)


@torch.no_grad()
def next_char_probs(model, vocab, prompt, k=5):
    """다음 글자 후보 상위 k개와 확률."""
    device = next(model.parameters()).device
    idx = torch.tensor([vocab.encode(prompt) or [vocab.end_id]], device=device)
    logits, _ = model(idx[:, -model.cfg.block_size:])
    probs = F.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, k)
    return [(vocab.chars[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
