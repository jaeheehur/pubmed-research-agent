"""
HuggingFace Hub에서 다운로드된 GGUF 모델 스캔
"""

import os
from pathlib import Path

def scan_huggingface_cache():
    """HuggingFace 캐시에서 GGUF 모델 찾기"""

    # HuggingFace 캐시 경로
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        print("HuggingFace 캐시 디렉토리를 찾을 수 없습니다.")
        return []

    print(f"캐시 디렉토리 스캔 중: {cache_dir}")
    print()

    gguf_models = []

    # models-- 로 시작하는 디렉토리 탐색
    for model_dir in cache_dir.glob("models--*"):
        # snapshots 디렉토리 확인
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        # 각 스냅샷 확인
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue

            # .gguf 파일 찾기
            for gguf_file in snapshot.glob("*.gguf"):
                # 모델 정보 추출
                model_name = model_dir.name.replace("models--", "").replace("--", "/")

                gguf_models.append({
                    "model_name": model_name,
                    "filename": gguf_file.name,
                    "path": str(gguf_file),
                    "size": gguf_file.stat().st_size / (1024**3),  # GB
                })

    return gguf_models


if __name__ == "__main__":
    print("="*80)
    print("설치된 GGUF 모델 목록")
    print("="*80)
    print()

    models = scan_huggingface_cache()

    if not models:
        print("설치된 GGUF 모델이 없습니다.")
        print()
        print("모델을 다운로드하려면:")
        print("  python download_gguf_model.py")
    else:
        print(f"총 {len(models)}개의 GGUF 모델 발견:")
        print()

        for i, model in enumerate(models, 1):
            print(f"{i}. {model['model_name']}")
            print(f"   파일: {model['filename']}")
            print(f"   크기: {model['size']:.2f} GB")
            print(f"   경로: {model['path']}")
            print()

        # 첫 번째 모델 경로를 model_path.txt에 저장
        if models:
            with open("model_path.txt", "w") as f:
                f.write(models[0]["path"])
            print(f"첫 번째 모델 경로가 'model_path.txt'에 저장되었습니다.")
            print()
            print("사용법:")
            print(f"  python agent_gguf.py $(cat model_path.txt)")
