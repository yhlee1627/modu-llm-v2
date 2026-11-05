"""둘째 마당: 이야기 LLM 놀이터(Gradio 화면)와 학습 단계 비교.

    from modu_llm import open_playground
    open_playground()                  # 코랩 셀 아래에 놀이터 화면이 열립니다.
"""
from .explore import STAGE_NAMES, load_story_llm, pad, show_char

__all__ = ["compare_stages", "open_playground"]

# 놀이터에서 고를 수 있는 모델(학습 단계 순서)
PLAY_STAGES = ["before", "step250", "step1000", "step3000", "pretrain"]


def compare_stages(texts, stages=PLAY_STAGES):
    """같은 문장들의 손실을 학습 단계마다 보여 줍니다(학습할수록 손실이 어떻게 바뀌는지)."""
    if isinstance(texts, str):
        texts = [texts]
    for i, t in enumerate(texts, 1):
        print(f"문장 {i}: {t}")
    print(pad("학습 단계", 20) + "".join(f"{f'문장 {i}':>8}" for i in range(1, len(texts) + 1)))
    for stage in stages:
        llm = load_story_llm(stage, verbose=False)
        print(pad(STAGE_NAMES[stage], 20) + "".join(f"{llm.loss(t):10.2f}" for t in texts))


# ── 놀이터 화면에서 부르는 계산 (화면 없이도 시험할 수 있게 따로 둠) ──

def play_next(llm, text, k=10, temperature=1.0, top_k=0):
    """다음 글자 후보: {글자: 확률} (Gradio Label에 바로 넣는 모양)."""
    rows = llm.top_chars(text, int(k), float(temperature), int(top_k) or None)
    return {f"{show_char(c)}": round(p, 4) for c, p in rows}


def play_attention(llm, text, layer=1, head=1):
    """비율 표(색칠한 HTML)와 글자마다 가장 많이 참고하는 글자 요약."""
    table = llm.attention(text, int(layer), int(head), total=True)
    chars = llm._chars(text)
    w = llm.attention_weights(text, int(layer), int(head))
    lines = []
    for t in range(len(chars)):
        j = int(w[t].argmax())
        lines.append(f"{t + 1}:{show_char(chars[t])} → {j + 1}:{show_char(chars[j])} ({w[t, j].item():.2f})")
    return table.to_html(), "\n".join(lines)


def play_losses(llm, text):
    """글자마다 [글자, 확률, 손실] 줄과 문장 손실."""
    rows = llm.char_losses(text)
    table = [[show_char(c), round(p, 3), round(l, 2)] for c, p, l in rows]
    mean = sum(r[2] for r in rows) / len(rows)
    return table, f"문장 손실(평균): {mean:.2f}"


def play_write(llm, text, length=100, temperature=0.8, top_k=0):
    """text에 이어 쓴 글(text 포함)."""
    return text + llm.continue_text(text, int(length), float(temperature), int(top_k) or None)


def open_playground(llm=None, share=None):
    """이야기 LLM 놀이터를 엽니다. 탭: 다음 글자, 비율 표, 손실, 이야기 쓰기.
    맨 위에서 학습 단계(학습 전 ~ 사전학습)를 고를 수 있습니다.
    share: 비워 두면 Gradio가 알아서 정합니다(코랩에서는 셀 아래에 화면이 열립니다)."""
    import gradio as gr

    cache = {}
    if llm is not None:
        cache[llm.stage] = llm

    def get(stage_name):
        stage = {STAGE_NAMES[s]: s for s in PLAY_STAGES}[stage_name]
        if stage not in cache:
            cache[stage] = load_story_llm(stage, verbose=False)
        return cache[stage]

    start = STAGE_NAMES[llm.stage] if llm is not None and llm.stage in PLAY_STAGES else STAGE_NAMES["pretrain"]
    with gr.Blocks(title="이야기 LLM 놀이터") as demo:
        gr.Markdown("## 이야기 LLM 놀이터\n미리 학습시켜 둔 이야기 LLM을 만져 봅니다. 글을 바꾸고 버튼을 누르세요.")
        stage = gr.Dropdown([STAGE_NAMES[s] for s in PLAY_STAGES], value=start, label="학습 단계")
        with gr.Tab("다음 글자"):
            t1 = gr.Textbox("어느 날, ", label="글")
            with gr.Row():
                k1 = gr.Slider(3, 20, value=10, step=1, label="후보 수")
                temp1 = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="온도")
                topk1 = gr.Slider(0, 20, value=0, step=1, label="top-k (0 = 쓰지 않음)")
            b1 = gr.Button("다음 글자 보기")
            o1 = gr.Label(label="다음 글자 후보와 확률")
            b1.click(lambda s, t, k, tp, tk: play_next(get(s), t, k, tp, tk), [stage, t1, k1, temp1, topk1], o1)
        with gr.Tab("비율 표"):
            t2 = gr.Textbox("토끼가 숲에 가요.", label="글")
            with gr.Row():
                layer2 = gr.Slider(1, 6, value=1, step=1, label="층")
                head2 = gr.Slider(1, 4, value=2, step=1, label="헤드")
            b2 = gr.Button("비율 표 보기")
            o2 = gr.HTML()
            o2b = gr.Textbox(label="글자마다 가장 많이 참고하는 글자", lines=8)
            b2.click(lambda s, t, l, h: play_attention(get(s), t, l, h), [stage, t2, layer2, head2], [o2, o2b])
        with gr.Tab("손실"):
            t3 = gr.Textbox("토끼가 숲에 가요.", label="글")
            b3 = gr.Button("손실 재기")
            o3 = gr.Dataframe(headers=["글자", "확률", "손실"], label="글자마다 손실")
            o3b = gr.Textbox(label="문장 손실")
            b3.click(lambda s, t: play_losses(get(s), t), [stage, t3], [o3, o3b])
        with gr.Tab("이야기 쓰기"):
            t4 = gr.Textbox("어느 날, ", label="시작 글")
            with gr.Row():
                len4 = gr.Slider(20, 300, value=100, step=10, label="글자 수")
                temp4 = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="온도")
                topk4 = gr.Slider(0, 50, value=0, step=1, label="top-k (0 = 쓰지 않음)")
            b4 = gr.Button("이야기 쓰기")
            o4 = gr.Textbox(label="이야기", lines=8)
            b4.click(lambda s, t, n, tp, tk: play_write(get(s), t, n, tp, tk), [stage, t4, len4, temp4, topk4], o4)
    if share is None:
        demo.launch()
    else:
        demo.launch(share=share)
    return demo
