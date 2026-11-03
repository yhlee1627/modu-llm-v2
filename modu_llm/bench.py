"""8장: 학습 한 단계에 걸리는 시간 재기."""
import time

import torch

from .data import get_batch


def time_steps(model, data, n=10, amp=False, batch_size=32):
    """학습 n단계에 걸리는 평균 시간(밀리초)을 돌려줍니다.
    처음 2단계는 준비 운동으로 보고 시간에서 뺍니다.
    amp=True면 반정밀도(autocast + GradScaler)로 계산합니다."""
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    use_amp = amp and on_gpu
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start = time.time()
    for i in range(n + 2):
        if i == 2:
            if on_gpu:
                torch.cuda.synchronize()   # GPU가 앞의 일을 다 끝낼 때까지 기다리기
            start = time.time()
        x, y = get_batch(data, model.cfg.block_size, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if on_gpu:
        torch.cuda.synchronize()
    return (time.time() - start) / n * 1000
