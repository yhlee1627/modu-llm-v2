"""숫자 표를 색칠해 보여 주는 도우미 (8~12장).

    show(cards, rows=["토", "끼", "가"])                  # 줄 이름만
    show(scores, rows=names, cols=names, mark_rows=2)     # 셋째 줄(번호 2)에 테두리
"""

# 색: 양수는 파랑, 음수는 주황. 크기가 클수록 진합니다.
_BLUE = (31, 119, 180)
_ORANGE = (230, 120, 20)
_ZERO = "#eeeeee"        # 정확히 0인 칸: 연한 회색
_MINUS_INF = "#bdbdbd"   # -inf(가린 칸): 회색


def _label(x):
    """칸 이름에서 눈에 보이지 않는 글자를 보이게 바꿉니다."""
    s = str(x)
    return {" ": "␣", "\n": "↵"}.get(s, s)


def _as_list(m, n, what):
    """테두리 번호를 목록으로. 텐서·numpy 숫자도 받고, -1처럼 뒤에서 세는 번호도 받습니다."""
    if m is None:
        return []
    if hasattr(m, "tolist"):
        m = m.tolist()
    items = m if isinstance(m, (list, tuple)) else [m]
    out = []
    for i in items:
        i = int(i)
        if not -n <= i < n:
            raise ValueError(f"{what} 번호 {i}는 표 밖입니다(0~{n - 1}).")
        out.append(i % n)
    return out


def _to_table(x):
    """텐서(또는 텐서 목록)를 (줄, 칸) 2차원 텐서로 바꿉니다."""
    import torch
    if isinstance(x, (list, tuple)) and all(isinstance(t, (int, float, bool)) for t in x):
        x = torch.tensor(x)                     # 숫자 목록은 한 줄로
    if isinstance(x, (list, tuple)):
        parts = [torch.as_tensor(t).detach().cpu().reshape(-1) for t in x]
        if any(p.is_floating_point() for p in parts):
            parts = [p.float() for p in parts]
        x = torch.stack(parts)
    t = torch.as_tensor(x).detach().cpu()
    if t.dim() == 0:
        return t.reshape(1, 1), 0
    if t.dim() == 1:
        return t.reshape(1, -1), 1
    if t.dim() == 2:
        return t, 2
    raise ValueError(f"show는 2차원 표까지 보여 줍니다. 이 텐서의 모양은 {tuple(t.shape)}입니다. "
                     "x[0]처럼 하나를 꺼내서 보여 주세요.")


def _cell_style(v, biggest):
    import math
    if v == float("-inf"):
        return f"background-color: {_MINUS_INF}; color: #333333"
    if v == 0 or biggest == 0 or math.isnan(v):
        return f"background-color: {_ZERO}; color: #333333"
    r, g, b = _BLUE if v > 0 else _ORANGE
    alpha = 0.12 + 0.78 * min(abs(v) / biggest, 1.0)
    text = "white" if alpha > 0.6 else "#222222"
    return f"background-color: rgba({r}, {g}, {b}, {alpha:.2f}); color: {text}"


def show(x, rows=None, cols=None, mark_rows=None, mark_cols=None, **old_names):
    """숫자 표를 색칠해서 바로 보여 줍니다.
    x: 텐서(숫자 하나, 한 줄, 2차원 표) 또는 한 줄짜리 텐서들의 목록
    rows, cols: 줄 이름과 칸 이름(비우면 0, 1, 2, …). 띄어쓰기는 ␣로 보입니다.
    mark_rows, mark_cols: 테두리를 칠 줄·칸의 번호(0부터). 숫자 하나나 목록.
    소수는 둘째 자리까지 보이고, 양수는 파랑, 음수는 주황, 0은 연한 회색, -inf는 회색입니다."""
    # 예전 이름(mark_row, mark_col)도 받습니다.
    mark_rows = old_names.pop("mark_row", mark_rows)
    mark_cols = old_names.pop("mark_col", mark_cols)
    if old_names:
        raise TypeError(f"show가 모르는 이름입니다: {', '.join(old_names)}")

    import pandas as pd
    t, dim = _to_table(x)
    n_rows, n_cols = t.shape
    row_names = [_label(r) for r in rows] if rows is not None else list(range(n_rows))
    col_names = [_label(c) for c in cols] if cols is not None else list(range(n_cols))
    if len(row_names) != n_rows or len(col_names) != n_cols:
        raise ValueError(f"표는 {n_rows}줄 × {n_cols}칸인데 줄 이름 {len(row_names)}개, "
                         f"칸 이름 {len(col_names)}개를 받았습니다.")
    is_float = t.is_floating_point()
    values = t.float().numpy() if is_float else t.numpy()
    hide_rows = dim == 0 or (dim == 1 and rows is None)
    hide_cols = dim == 0

    # 텍스트 표: 노트북 밖에서 실행할 때(그리고 책의 실행 결과). 한글은 두 칸으로 셈해 줄을 맞춥니다.
    def text():
        import unicodedata
        wid = lambda s: sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
        cell = (lambda v: "-inf" if v == float("-inf") else f"{v:.2f}") if is_float else str
        body = [[cell(v) for v in row] for row in values.tolist()]
        heads = [str(c) for c in col_names]
        widths = [max([wid(heads[j]) if not hide_cols else 0] + [wid(r[j]) for r in body]) for j in range(n_cols)]
        left = 0 if hide_rows else max(wid(str(r)) for r in row_names)
        lines = []
        if not hide_cols:
            lines.append(" " * left + "".join(" " * (widths[j] + 2 - wid(heads[j])) + heads[j] for j in range(n_cols)))
        for i, r in enumerate(body):
            name = "" if hide_rows else str(row_names[i]) + " " * (left - wid(str(row_names[i])))
            lines.append(name + "".join(" " * (widths[j] + 2 - wid(r[j])) + r[j] for j in range(n_cols)))
        return "\n".join(line.rstrip() for line in lines)

    try:
        from IPython import get_ipython
        from IPython.display import display
        in_notebook = get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        in_notebook = False
    _as_list(mark_rows, n_rows, "줄"), _as_list(mark_cols, n_cols, "칸")   # 번호가 표 안인지 먼저 확인
    if not in_notebook:
        print(text())
        return

    # 색칠한 표: 같은 이름의 줄·칸(예: 두 개의 '가')이 있어도 되도록 번호로 만든 뒤 이름을 붙입니다.
    import math
    df = pd.DataFrame(values)
    finite = [abs(v) for v in values.reshape(-1).tolist() if not (math.isinf(v) or math.isnan(v))]
    biggest = max(finite) if finite else 0
    marked_r, marked_c = set(_as_list(mark_rows, n_rows, "줄")), set(_as_list(mark_cols, n_cols, "칸"))
    line = "3px solid #222222"

    def styles(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for i in range(n_rows):
            for j in range(n_cols):
                s = _cell_style(float(values[i, j]), biggest) if is_float else "color: #222222"
                s += "; border: 1px solid #ffffff"
                if i in marked_r:
                    s += f"; border-top: {line}; border-bottom: {line}"
                    if j == 0:
                        s += f"; border-left: {line}"
                    if j == n_cols - 1:
                        s += f"; border-right: {line}"
                if j in marked_c:
                    s += f"; border-left: {line}; border-right: {line}"
                    if i == 0:
                        s += f"; border-top: {line}"
                    if i == n_rows - 1:
                        s += f"; border-bottom: {line}"
                out.iloc[i, j] = s
        return out

    st = df.style.apply(styles, axis=None)
    st = st.format((lambda v: "-inf" if v == float("-inf") else f"{v:.2f}") if is_float else "{}")
    import html
    esc = lambda names: [html.escape(str(v)) for v in names]   # <s> 같은 이름도 글자 그대로 보이게
    st = st.relabel_index(esc(row_names), axis=0).relabel_index(esc(col_names), axis=1)
    st = st.set_table_styles([
        {"selector": "", "props": "border-collapse: collapse"},
        {"selector": "td", "props": "text-align: center; min-width: 3.2em; padding: 4px 6px; "
                                    "font-family: monospace"},
        {"selector": "th", "props": "text-align: center; padding: 4px 6px; font-weight: normal"},
    ])
    if hide_rows:
        st = st.hide(axis=0)
    if hide_cols:
        st = st.hide(axis=1)
    display(st)


def show_weights(w, labels, digits=2):
    """어텐션 비율 표(pandas 표): 줄 = 지금 글자, 세로줄 = 참고하는 글자.
    w는 (글자, 글자) 모양이고, (배치, 글자, 글자)이면 첫 번째만 보여 줍니다."""
    import pandas as pd
    if w.dim() == 3:
        w = w[0]
    table = w.detach().cpu().numpy().round(digits)
    return pd.DataFrame(table, index=labels, columns=labels)
