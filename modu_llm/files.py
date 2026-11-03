"""모든 장: 저장물 찾기. 독자가 만든 파일이 있으면 그것을, 없으면 저자 파일을 내려받습니다."""
import os

from .settings import ASSET_REPO, DRIVE_DIR


def mount_drive():
    """구글 드라이브를 연결하고 작업 폴더를 만듭니다. (Colab 전용)"""
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    return DRIVE_DIR


def get_file(name, folder=DRIVE_DIR, repo=ASSET_REPO):
    """name 파일의 경로를 돌려줍니다.
    ① 드라이브 작업 폴더 ② 현재 폴더 ③ 저자 허깅페이스 순서로 찾습니다."""
    for d in [folder, "."]:
        path = os.path.join(d, name)
        if os.path.exists(path):
            print(f"내 파일 사용: {path}")
            return path
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, name)
    print(f"저자 파일 내려받음: {name}")
    return path
