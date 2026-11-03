"""2·3·9장: 이야기 불러오기, 인코딩, 배치."""
import os
import time

import numpy as np
import torch

from .settings import STORY_REPO


def load_stories(n, skip=0, repo=STORY_REPO):
    """이야기 n편을 스트리밍으로 읽습니다."""
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    if skip:
        ds = ds.skip(skip)
    return [row["text"].strip() for row in ds.take(n)]


def encode_texts(texts, vocab):
    """글마다 인코딩하고 끝 표시를 붙여 하나의 uint16 배열로 이어 붙입니다.
    파이썬 리스트 하나에 모두 담지 않아 메모리를 아낍니다."""
    end = vocab.end_id
    parts = [np.array(vocab.encode(t) + [end], dtype=np.uint16) for t in texts]
    return np.concatenate(parts)


def split_data(data, ratio=0.9):
    n = int(len(data) * ratio)
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, device="cpu"):
    """임의 위치에서 글 조각을 잘라 입력 x와 정답 y(한 칸 뒤)를 만듭니다."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def prepare_stories(n, vocab, folder, ratio=0.9):
    """9장: 이야기 n편을 번호 배열로 준비하고 (학습용, 검증용)을 돌려줍니다.
    folder에 tokens_n.npy가 있으면 불러오고, 없으면 내려받아 인코딩한 뒤 저장합니다."""
    path = os.path.join(folder, f"tokens_{n}.npy")
    if os.path.exists(path):
        data = np.load(path)
        print("저장된 데이터를 불러왔습니다.")
    else:
        t0 = time.time()
        stories = load_stories(n)
        data = encode_texts(stories, vocab)
        np.save(path, data)
        print(f"이야기 {len(stories):,}편 인코딩 완료 ({time.time() - t0:.0f}초)")
    print(f"글자 수: {len(data):,}")
    print(f"uint16 배열: {round(data.nbytes / 1e6):,}MB")
    print(f"파이썬 정수 리스트였다면 약 {round(len(data) * 36 / 1e6):,}MB")
    return split_data(data, ratio)
