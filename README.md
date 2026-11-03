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
| `llm.losses(text)` | 글자마다 손실과 문장 손실 |
| `llm.attention(text, layer, head)` | 어텐션 비율 표 |

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
