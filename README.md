# 모두의 LLM with 파이썬 (개편판) — 실습 코드

LLM의 원리를 그림과 비유로 배우고, 미리 학습시켜 둔 작은 이야기 LLM과 허깅페이스의 진짜 LLM을 직접 실행하며 확인하는 실습 노트북과 패키지입니다. 모든 노트북은 Google Colab에서 실행합니다.

## 책의 구성

| 마당 | 장 | 실습에서 하는 일 |
|---|---|---|
| 첫째 LLM 첫걸음 | 1~2장 | 이야기 LLM을 불러와 다음 글자 후보를 보고 이야기를 이어 씀 |
| 둘째 LLM 원리 놀이터 | 3~7장 | 이야기 LLM 안을 들여다봄(어텐션 비율 표, 문장 손실, 온도) |
| 셋째 나만의 작은 GPT 만들기 | 8~12장 | 텐서로 어텐션을 직접 계산하고, 작은 GPT를 조립해 학습함 |
| 넷째 진짜 LLM 들여다보기 | 13~14장 | HyperCLOVAX SEED 0.5B와 Gemma 3 1B를 이야기 LLM과 같은 방법으로 들여다봄 |

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
| 8 | `notebooks/ch08.ipynb` 이야기 데이터를 숫자 표로: 텐서와 임베딩 | 불필요 |
| 9 | `notebooks/ch09.ipynb` 어텐션 헤드 직접 계산하기 | 불필요 |
| 10 | `notebooks/ch10.ipynb` 작은 GPT 조립하기 | 불필요 |
| 11 | `notebooks/ch11.ipynb` 사전학습과 글 만들기 | 30분 학습에 필요(T4) |
| 12 | `notebooks/ch12.ipynb` 파인튜닝과 나만의 챗봇 | 파인튜닝에 필요(T4) |
| 13 | `notebooks/ch13.ipynb` 진짜 LLM에 같은 질문 던지기 | 불필요(내려받기 약 1.1GB) |
| 14 | `notebooks/ch14.ipynb` 진짜 LLM 속 들여다보기 | 불필요(Gemma는 약관 동의와 토큰 필요, 부록 C) |
| 부록 A | `notebooks/appA.ipynb` 파이썬 기초 | 불필요 |

11·12장의 학습 셀은 건너뛰어도 됩니다. 다음 셀에서 미리 학습시켜 둔 모델 파일을 내려받아 이어 갑니다.

## 패키지 `modu_llm`

```python
%pip install -q git+https://github.com/yhlee1627/modu-llm-v2
from modu_llm import load_story_llm
llm = load_story_llm()          # 미리 학습시켜 둔 이야기 LLM
llm.next_chars("어느 날, ")      # 다음 글자 후보와 확률
llm.write("어느 날, ")           # 이어서 이야기 쓰기
```

| 1~7장에서 쓰는 것 | 하는 일 |
|---|---|
| `load_story_llm(stage)` | 이야기 LLM 불러오기. `"pretrain"`(사전학습), `"sft"`(파인튜닝), `"feedback"`(피드백 학습), `"before"`(학습 전), `"step250"`·`"step1000"`·`"step3000"`(사전학습 도중) |
| `llm.next_chars(text, k=5)` | 다음 글자 후보 k개와 확률 |
| `llm.write(text, length=100, temperature=0.8, top_k=None, n=1)` | 이어서 글 쓰기 |
| `llm.ask(request)` | 요청을 넣고 이야기 받기(파인튜닝한 모델) |
| `llm.to_ids(text)`, `llm.to_text(ids)` | 글자 ↔ 번호 |
| `llm.cards(text)` | 글자 카드 + 자리 카드 = 합친 카드 |
| `llm.compare_tokens(text, *names)` | 허깅페이스의 다른 LLM들과 토큰 나누기 비교 |
| `llm.attention(text, layer, head, total=False)` | 어텐션 비율 표 |
| `llm.attention_steps(text, layer, head)` | 글자 하나의 점수 → 나누기 → 가리기 → 비율 |
| `llm.looks_at(text, layer, head)` | 글자마다 가장 많이 참고하는 글자 |
| `llm.heads_summary(text, top=None, by=None)` | 헤드 24개의 성향 요약(top·by를 주면 by 칸이 큰 헤드 top개) |
| `llm.layer_guess(text)` | 층마다 다음 글자 맞히기 |
| `llm.info()` | 모델 크기와 파라미터 수 |
| `llm.losses(text, repeat=1)`, `llm.compare_losses(texts)` | 글자마다 손실(repeat=2면 두 번 넣고 비교), 문장마다 손실 |
| `compare_stages(texts)` | 학습 단계별 문장 손실 |
| `open_playground()` | 이야기 LLM 놀이터(Gradio 화면) |

`llm.next_chars`와 `llm.write`에는 `temperature`(온도)와 `top_k`를 줄 수 있습니다.

| 8~12장에서 쓰는 것 | 하는 일 |
|---|---|
| `Vocab.build()` | 글자 사전 2,453자 |
| `load_stories(n)`, `encode_texts(texts, vocab)` | 이야기 내려받기, 번호 배열로 바꾸기 |
| `GPTConfig`, `Embeddings`, `Head`, `Block`, `TinyGPT` | 기본 GPT 부품(`model_basic.py`, 한국어 주석) |
| `show(x, rows, cols, mark_rows, mark_cols)` | 숫자 표를 색칠해 보기(양수 파랑, 음수 주황) |
| `get_batch(data, block_size, batch_size)` | 문제와 정답 조각 만들기 |
| `get_device()`, `mount_drive()`, `get_file(name)` | GPU 확인, 드라이브 연결, 미리 학습시켜 둔 모델 파일 내려받기 |
| `prepare_stories(n, vocab, folder)` | 사전학습용 데이터 준비 |
| `GPT`, `train(...)`, `load_checkpoint(...)`, `plot_history(history)` | 빠른 GPT(`model.py`), 학습 반복, 이어서 학습하기, 손실 그래프 |
| `generate(model, vocab, text, ...)` | 이어서 글 쓰기 |
| `format_pair(request)`, `load_sft_texts(split)` | 요청-이야기 형식, 파인튜닝 데이터 |
| `chat_app(chat_fn)` | 나만의 이야기 챗봇 화면(Gradio) |

| 13~14장에서 쓰는 것 | 하는 일 |
|---|---|
| `load_llm(name)` | 허깅페이스의 LLM 불러오기(예: `"naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"`) |
| `llm.info()`, `llm.next_chars(text)`, `llm.write(text, length)` | 크기, 다음 토큰 후보, 이어서 쓰기 |
| `llm.losses(text, repeat=1)`, `llm.compare_losses(texts)` | 토큰마다 손실, 문장마다 손실 |
| `llm.attention(text, layer, head)`, `llm.looks_at(text, layer, head)` | 비율 표, 토큰마다 가장 많이 참고하는 토큰 |
| `llm.heads_summary(text, top, by)` | 헤드 모두를 재고 by 칸(바로 앞, 따라 하기 등)이 큰 헤드 top개 |
| `llm.layer_guess(text)` | 층마다 다음 토큰 맞히기 |

## 데이터와 모델을 만든 노트북 (`extras/`)

책을 따라가는 데 필요하지 않습니다. 같은 방법을 직접 해 보고 싶을 때 참고하세요.

| 노트북 | 하는 일 |
|---|---|
| `extras/SFT_데이터_만들기.ipynb` | 12장 요청-이야기 데이터 만들기 |
| `extras/피드백_모델_만들기.ipynb` | 2장 피드백 모델: 규칙으로 고른 선호 데이터 짝 920개 + DPO |

## 모델 설정

글자 사전 2,453자(한글 완성형 2,350자 + 영문·숫자·기호 + 끝 표시 `§`) · 문맥 384글자 · 벡터 길이 256 · 헤드 4개 · 블록 6층 · 약 600만 개 파라미터

이야기 LLM 체크포인트: [huggingface.co/JosephLee/modu-llm](https://huggingface.co/JosephLee/modu-llm) (CC BY 4.0)

## 데이터 출처

- [Tiny-Ko-Stories](https://huggingface.co/datasets/psymon/Tiny-Ko-Stories) (psymon, CC BY 4.0)
- [Tiny-Ko-Stories SFT](https://huggingface.co/datasets/JosephLee/tiny-ko-stories-sft): 요청-이야기 데이터, Tiny-Ko-Stories를 가공 (CC BY 4.0)
- 13·14장 모델: [HyperCLOVAX SEED 0.5B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B), [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-pt) — 각 모델의 라이선스를 따르며 이 저장소에 포함하지 않습니다.
- 모델 구조 참고: [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)

## 라이선스

코드는 MIT 라이선스(`LICENSE`)를 따릅니다. 이야기 LLM 체크포인트와 데이터는 CC BY 4.0이며, 쓸 때 위 출처를 함께 밝혀 주세요.
