"""11·12장: 요청-이야기 형식."""
from .settings import SFT_REPO


def format_pair(request, story=""):
    """요청과 이야기를 하나의 글로. 끝 표시(§)는 encode_texts가 붙입니다.
    story를 비우면 챗봇에 넣을 질문 형태가 됩니다."""
    return f"요청: {request}\n이야기: {story}"


def load_sft_texts(split="train", repo=SFT_REPO, limit=None):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_pair(r["request"], r["story"]) for r in ds]
