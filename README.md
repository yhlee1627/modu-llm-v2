# 모두의 LLM with 파이썬 (개편판) — 실습 코드

LLM의 원리를 그림과 비유로 배우고, 저자가 학습시킨 작은 이야기 LLM을 직접 실행하며 확인하는 실습 노트북과 패키지입니다. 모든 노트북은 Google Colab에서 실행합니다.

## 책의 구성

| 마당 | 장 | 실습에서 하는 일 |
|---|---|---|
| 첫째 LLM 첫걸음 | 1~2장 | 이야기 LLM을 불러와 다음 글자 후보를 보고 이야기를 이어 씀 |
| 둘째 LLM 원리 놀이터 | 3~7장 | 이야기 LLM 안을 들여다봄(어텐션 비율 표, 문장 손실, 온도) |
| 셋째 LLM 계산 도구 상자 | 8~10장 | 텐서 계산을 몇 줄씩 직접 씀 |
| 넷째 나만의 LLM 만들기 | 11~14장 | 작은 GPT를 조립하고 학습함 |

## 장별 노트북

| 장 | 노트북 | GPU |
|---|---|---|
| 1 | `notebooks/ch01.ipynb` LLM이란 무엇일까요? | 불필요 |
| 2 | `notebooks/ch02.ipynb` LLM은 어떻게 똑똑해질까요? | 불필요 |
| 3 | `notebooks/ch03.ipynb` 글자를 숫자로 바꾸기: 토큰과 임베딩 | 불필요 |
| 4 | `notebooks/ch04.ipynb` 앞의 글자를 참고하는 어텐션 | 불필요 |
| 5 | `notebooks/ch05.ipynb` 여러 눈으로 보고 혼자 정리하기: Transformer 블록 | 불필요 |
| 6 | `notebooks/ch06.ipynb` LLM은 어떻게 배우고 말할까요? | 불필요 |
| 7 | `notebooks/ch07.ipynb` LLM 놀이터: 내 이야기 LLM 들여다보기 | 불필요 |

(장을 마칠 때마다 추가합니다.)

## 패키지 `modu_llm`

```python
%pip install -q git+https://github.com/yhlee1627/modu-llm-v2
from modu_llm import load_story_llm
llm = load_story_llm()          # 저자가 학습시킨 이야기 LLM
llm.next_chars("어느 날, ")      # 다음 글자 후보와 확률
llm.write("어느 날, ")           # 이어서 이야기 쓰기
```

| 1~7장에서 쓰는 것 | 하는 일 |
|---|---|
| `load_story_llm(stage)` | 이야기 LLM 불러오기. `"pretrain"`(사전학습), `"sft"`(파인튜닝), `"before"`(학습 전), `"step250"`·`"step1000"`·`"step3000"`(사전학습 도중) |
| `llm.next_chars(text, k=5)` | 다음 글자 후보 k개와 확률 |
| `llm.write(text, length=100, temperature=0.8, top_k=None, n=1)` | 이어서 글 쓰기 |
| `llm.ask(request)` | 요청을 넣고 이야기 받기(파인튜닝한 모델) |
| `llm.to_ids(text)`, `llm.to_text(ids)` | 글자 ↔ 번호 |
| `llm.cards(text)` | 글자 카드 + 자리 카드 = 합친 카드 |
| `llm.compare_tokens(text, name)` | 허깅페이스의 다른 LLM과 토큰 나누기 비교 |
| `llm.attention(text, layer, head, total=False)` | 어텐션 비율 표 |
| `llm.attention_steps(text, layer, head)` | 글자 하나의 점수 → 나누기 → 가리기 → 비율 |
| `llm.looks_at(text, layer, head)` | 글자마다 가장 많이 참고하는 글자 |
| `llm.heads_summary(text)` | 헤드 24개의 성향 요약 |
| `llm.layer_guess(text)` | 층마다 다음 글자 맞히기 |
| `llm.info()` | 모델 크기와 파라미터 수 |
| `llm.losses(text)`, `llm.compare_losses(texts)` | 글자마다 손실, 문장마다 손실 |
| `compare_stages(texts)` | 학습 단계별 문장 손실 |
| `open_playground()` | 이야기 LLM 놀이터(Gradio 화면) |

`llm.next_chars`와 `llm.write`에는 `temperature`(온도)와 `top_k`를 줄 수 있습니다.

8~14장에서 쓰는 부품(글자 사전, 임베딩, 어텐션, 블록, GPT, 학습)도 같은 패키지에 들어 있습니다. 모델 코드는 `modu_llm/model_basic.py`, `modu_llm/model.py`에서 볼 수 있습니다.

## 모델 설정

글자 사전 2,453자(한글 완성형 2,350자 + 영문·숫자·기호 + 끝 표시 `§`) · 문맥 384글자 · 벡터 길이 256 · 헤드 4개 · 블록 6층 · 약 600만 개 파라미터

저자 체크포인트: [huggingface.co/JosephLee/modu-llm](https://huggingface.co/JosephLee/modu-llm) (CC BY 4.0)

## 데이터 출처

- [Tiny-Ko-Stories](https://huggingface.co/datasets/psymon/Tiny-Ko-Stories) (psymon, CC BY 4.0)
- [Tiny-Ko-Stories SFT](https://huggingface.co/datasets/JosephLee/tiny-ko-stories-sft): 요청-이야기 데이터, Tiny-Ko-Stories를 가공 (CC BY 4.0)
- 모델 구조 참고: [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)

## 라이선스

코드는 MIT 라이선스(`LICENSE`)를 따릅니다. 저자 체크포인트와 데이터는 CC BY 4.0이며, 쓸 때 위 출처를 함께 밝혀 주세요.
