"""여러 장에서 되풀이되는 준비·그래프 코드를 묶은 도우미."""
import torch


def get_device(seed=42, need_gpu=False):
    """난수 시드를 정하고, GPU가 있으면 "cuda", 없으면 "cpu"를 돌려줍니다.
    need_gpu=True인데 GPU가 없으면 런타임을 바꾸는 방법을 알려 줍니다."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("장치:", device)
    if need_gpu and device == "cpu":
        print("⚠ GPU가 연결되지 않았어요. 런타임 → 런타임 유형 변경 → T4 GPU를 고르세요.")
        print("  GPU를 받을 수 없다면 이 장의 학습은 건너뛰고, 다음 장에서 저자 체크포인트를 쓰면 돼요.")
    return device


def plot_history(history, mark_step=None):
    """손실 기록(history)을 그래프로 그립니다.
    mark_step을 주면 그 단계에 세로 점선을 긋습니다."""
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    plt.plot(steps, [h["train"] for h in history], label="train")
    plt.plot(steps, [h["val"] for h in history], label="validation")
    if mark_step is not None:
        plt.axvline(mark_step, color="gray", linestyle="--")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
