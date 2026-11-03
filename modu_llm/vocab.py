"""2장: 글자 사전."""
import json

from .settings import END


def hangul_2350():
    """자주 쓰는 한글 완성형 2,350자 (KS X 1001 순서)."""
    return [bytes([hi, lo]).decode("euc-kr")
            for hi in range(0xB0, 0xC9) for lo in range(0xA1, 0xFF)]


EXTRA = list("…‘’“”·")   # 이야기에 자주 나오는 기호


def build_chars():
    """고정 글자 목록: 줄바꿈 + 영문·숫자·기호 + 추가 기호 + 한글 2,350자 + 끝 표시."""
    ascii_chars = [chr(c) for c in range(32, 127)]
    return ["\n"] + ascii_chars + EXTRA + hangul_2350() + [END]


class Vocab:
    """글자와 번호를 오가는 사전."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    @property
    def end_id(self):
        return self.stoi[END]

    def encode(self, s):
        """사전에 없는 글자는 건너뜁니다."""
        ids = []
        for ch in s:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            chars.append(self.chars[int(i)])
        return "".join(chars)

    def unknown_chars(self, s):
        """사전에 없는 글자 목록."""
        return sorted(set(s) - set(self.stoi))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def build(cls):
        return cls(build_chars())
