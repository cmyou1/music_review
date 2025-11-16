"""
Download HTSAT/MERT checkpoint locally for offline use.
"""

from pathlib import Path

from huggingface_hub import snapshot_download


def download(model_id: str = "m-a-p/MERT-v1-95M", target_dir: str = "./models/htsat"):
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=path, local_dir_use_symlinks=False)
    print(f"모델 다운로드 완료: {path.resolve()}")


if __name__ == "__main__":
    download()

