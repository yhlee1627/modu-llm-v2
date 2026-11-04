"""『모두의 LLM with 파이썬』 실습 패키지.
1~7장: 미리 학습시켜 둔 이야기 LLM 들여다보기(load_story_llm)
8~12장: 작은 GPT를 만들고 학습하는 부품과 도우미
13~14장: 허깅페이스의 진짜 LLM 들여다보기(load_llm)"""
from .settings import ASSET_REPO, DRIVE_DIR, END, SFT_REPO, STORY_REPO
from .vocab import Vocab, build_chars, hangul_2350
from .data import encode_texts, get_batch, load_stories, prepare_stories, split_data
from .model_basic import Block, Embeddings, GPTConfig, Head, MultiHead, TinyGPT
from .model import GPT, CausalSelfAttention, FastBlock, copy_basic_to_fast, count_params
from .generate import generate, next_char_probs
from .train import estimate_loss, load_checkpoint, save_checkpoint, train
from .sft import format_pair, load_sft_texts
from .files import get_file, mount_drive
from .viz import show, show_weights
from .bench import time_steps
from .helpers import get_device, plot_history
from .explore import StoryLLM, load_story_llm
from .real import RealLLM, load_llm
from .playground import *  # noqa: F401,F403  (7장)
from .chat import *  # noqa: F401,F403  (12장)
from . import playground as _pg, chat as _chat

__version__ = "0.4.0"

__all__ = [
    "ASSET_REPO", "DRIVE_DIR", "END", "SFT_REPO", "STORY_REPO",
    "Vocab", "build_chars", "hangul_2350", "encode_texts", "get_batch", "load_stories", "prepare_stories", "split_data",
    "Block", "Embeddings", "GPTConfig", "Head", "MultiHead", "TinyGPT",
    "GPT", "CausalSelfAttention", "FastBlock", "copy_basic_to_fast", "count_params",
    "generate", "next_char_probs",
    "estimate_loss", "load_checkpoint", "save_checkpoint", "train",
    "format_pair", "load_sft_texts", "get_file", "mount_drive", "show", "show_weights", "time_steps",
    "get_device", "plot_history",
    "StoryLLM", "load_story_llm", "RealLLM", "load_llm",
] + list(_pg.__all__) + list(_chat.__all__)
