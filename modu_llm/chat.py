"""14장: 나만의 이야기 챗봇의 대화 화면(Gradio).

독자는 요청을 받아 이야기를 돌려주는 chat 함수를 직접 만들고,
화면은 이 파일의 chat_app 한 줄로 띄웁니다.
    demo = chat_app(chat)
    demo.launch()
"""

__all__ = ["chat_app"]

EXAMPLE = "여름날 바닷가에서 토끼가 나오는 이야기를 들려줘"


def chat_app(chat_fn, title="나만의 이야기 챗봇", example=EXAMPLE):
    """chat_fn(요청, 온도) → 이야기 를 대화 화면으로 감쌉니다.
    화면에는 요청 칸, 온도 막대, 이야기 칸이 있습니다. 돌려주는 demo에서 demo.launch()로 띄웁니다."""
    import gradio as gr

    request_box = gr.Textbox(label="요청", value=example)
    temp_slider = gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="온도")
    story_box = gr.Textbox(label="이야기", lines=12)
    return gr.Interface(
        fn=chat_fn,
        inputs=[request_box, temp_slider],
        outputs=story_box,
        title=title,
        flagging_mode="never",      # 결과를 파일로 모으는 Flag 버튼은 쓰지 않음
    )
