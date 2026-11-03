"""어텐션 비율을 표로 보여 주는 도우미 (4~6장)."""


def show_weights(w, labels, digits=2):
    """어텐션 비율 표: 줄 = 지금 글자, 세로줄 = 참고하는 글자.
    w는 (글자, 글자) 모양이고, (배치, 글자, 글자)이면 첫 번째만 보여 줍니다."""
    import pandas as pd
    if w.dim() == 3:
        w = w[0]
    table = w.detach().cpu().numpy().round(digits)
    return pd.DataFrame(table, index=labels, columns=labels)
