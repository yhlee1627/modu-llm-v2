"""9장: 학습 루프와 체크포인트. 11장에서 그대로 다시 씁니다."""
import os
import time

import torch

from .data import get_batch
from .model import GPT
from .model_basic import GPTConfig


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size=32, iters=20):
    """학습용·검증용 데이터에서 손실 평균."""
    model.eval()
    device = next(model.parameters()).device
    result = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(iters):
            x, y = get_batch(data, model.cfg.block_size, batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def save_checkpoint(path, model, optimizer=None, step=0, history=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "cfg": dict(vars(model.cfg)),   # 설정 다섯 개를 딕셔너리로
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "history": history or [],
    }, path)


def load_checkpoint(path, device="cpu", optimizer_lr=None):
    """체크포인트에서 모델(과 원하면 옵티마이저)을 되살립니다.
    돌려주는 값: model, optimizer(또는 None), step, history"""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = None
    if optimizer_lr is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
            for g in optimizer.param_groups:
                g["lr"] = optimizer_lr
    return model, optimizer, ckpt.get("step", 0), ckpt.get("history", [])


def train(model, optimizer, train_data, val_data, *, minutes, ckpt_path,
          batch_size=32, eval_every=200, save_every_min=5, start_step=0, history=None,
          snapshots=None, vocab=None, sample_prompt="어느 날, ", keep_best=False,
          verbose=True, on_eval=None):
    """정해진 시간(분) 동안 학습하고 (마지막 단계 수, 손실 기록)을 돌려줍니다.
    - eval_every 단계마다 학습·검증 손실을 기록하고 출력합니다.
      vocab을 주면 그때마다 sample_prompt로 짧은 글도 만들어 보여 줍니다.
    - keep_best=False: save_every_min 분마다, 그리고 끝날 때 ckpt_path에 저장합니다.
    - keep_best=True: 검증 손실이 지금까지 중 가장 낮을 때만 ckpt_path에 저장합니다.
    - snapshots=[250, 1000]처럼 주면 그 단계가 되는 순간 ckpt_stepN.pt를 따로 남깁니다."""
    from .generate import generate
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if history is None:
        history = []
    if snapshots is None:
        snapshots = []
    folder = os.path.dirname(ckpt_path) or "."
    best = float("inf")
    for h in history:
        best = min(best, h["val"])
    step = start_step
    start = last_save = time.time()
    model.train()
    while time.time() - start < minutes * 60:
        if step % eval_every == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size)
            history.append({"step": step, "train": losses["train"], "val": losses["val"]})
            if verbose:
                elapsed = (time.time() - start) / 60
                print(f"step {step:5d} | 학습 {losses['train']:.3f} | 검증 {losses['val']:.3f} | {elapsed:4.1f}분")
                if vocab is not None:
                    sample = generate(model, vocab, sample_prompt, max_new_tokens=60)
                    print("    ", sample.replace("\n", " "))
            if on_eval:
                on_eval(step, losses)
            if keep_best and losses["val"] < best:
                best = losses["val"]
                save_checkpoint(ckpt_path, model, optimizer, step, history)
        x, y = get_batch(train_data, model.cfg.block_size, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step in snapshots:
            save_checkpoint(f"{folder}/ckpt_step{step}.pt", model, None, step, history)
        if not keep_best and time.time() - last_save > save_every_min * 60:
            save_checkpoint(ckpt_path, model, optimizer, step, history)
            last_save = time.time()
            if verbose:
                print("     (저장했습니다)")
    losses = estimate_loss(model, train_data, val_data, batch_size)
    history.append({"step": step, "train": losses["train"], "val": losses["val"]})
    if not keep_best or losses["val"] < best:
        save_checkpoint(ckpt_path, model, optimizer, step, history)
    return step, history
