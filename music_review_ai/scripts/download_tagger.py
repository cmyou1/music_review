"""
Download AudioSet AST tag classifier locally for offline use.
"""

from pathlib import Path

from huggingface_hub import snapshot_download


def download(model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593", target_dir: str = "./models/tagger"):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=target, local_dir_use_symlinks=False)
    print(f"태그 분류 모델 다운로드 완료: {target.resolve()}")


if __name__ == "__main__":
    download()
