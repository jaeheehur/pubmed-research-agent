"""
GGUF 모델 다운로드 스크립트 (설치된 모델 감지 포함)
"""

from huggingface_hub import hf_hub_download
import os
from pathlib import Path

def scan_installed_models():
    """설치된 GGUF 모델 스캔"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        return []

    gguf_models = []
    for model_dir in cache_dir.glob("models--*"):
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue

            for gguf_file in snapshot.glob("*.gguf"):
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                gguf_models.append({
                    "model_name": model_name,
                    "filename": gguf_file.name,
                    "path": str(gguf_file),
                    "size_gb": gguf_file.stat().st_size / (1024**3),
                })

    return gguf_models

print("="*80)
print("GGUF 모델 관리")
print("="*80)

# 시도할 모델 목록 (우선순위 순서)
models = [
    # 1. JSL-MedLlama-3-8B Q6_K (의료 특화, 가장 정확)
    {
        "name": "JSL-MedLlama-3-8B Q6_K (의료 특화, 권장)",
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v2.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v2.0-Q6_K.gguf",
        "size": "~6.59GB"
    },
    # 2. JSL-MedLlama-3-8B Q4_K_M (더 빠름)
    {
        "name": "JSL-MedLlama-3-8B Q4_K_M (빠른 버전)",
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v2.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v2.0-Q4_K_M.gguf",
        "size": "~4.92GB"
    },
    # 3. BioMistral (다른 저장소)
    {
        "name": "BioMistral-7B Q4",
        "repo_id": "MaziyarPanahi/BioMistral-7B-GGUF",
        "filename": "BioMistral-7B.Q4_K_M.gguf",
        "size": "~4GB"
    },
    # 4. Llama-3.2 의료 특화 버전
    {
        "name": "Llama-3.2-1B Medical Q4",
        "repo_id": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size": "~600MB"
    },
    # 5. Mistral 7B (범용, 안정적)
    {
        "name": "Mistral-7B Q4",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "~4GB"
    },
    # 6. TinyLlama (작고 빠름)
    {
        "name": "TinyLlama-1.1B Q4 (가장 빠름)",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "~600MB"
    }
]

# 설치된 모델 확인
print("\n설치된 GGUF 모델 확인 중...")
installed_models = scan_installed_models()

if installed_models:
    print(f"\n이미 설치된 모델 ({len(installed_models)}개):")
    for i, model in enumerate(installed_models, 1):
        print(f"  [{i}] {model['model_name']}")
        print(f"      파일: {model['filename']}")
        print(f"      크기: {model['size_gb']:.2f} GB")
        print()

    use_installed = input("설치된 모델을 사용하시겠습니까? (y/n) [기본값: y]: ").strip().lower()

    if use_installed != 'n':
        # 설치된 모델 선택
        if len(installed_models) == 1:
            selected_path = installed_models[0]['path']
            print(f"\n선택: {installed_models[0]['filename']}")
        else:
            idx_choice = input(f"모델을 선택하세요 (1-{len(installed_models)}) [기본값: 1]: ").strip() or "1"
            try:
                idx = int(idx_choice) - 1
                if 0 <= idx < len(installed_models):
                    selected_path = installed_models[idx]['path']
                    print(f"\n선택: {installed_models[idx]['filename']}")
                else:
                    selected_path = installed_models[0]['path']
                    print(f"\n선택: {installed_models[0]['filename']}")
            except:
                selected_path = installed_models[0]['path']
                print(f"\n선택: {installed_models[0]['filename']}")

        # 경로 저장
        with open("model_path.txt", "w") as f:
            f.write(selected_path)

        print(f"\n✓ 모델 경로가 'model_path.txt'에 저장되었습니다.")
        print(f"\n사용법:")
        print(f"  python agent_gguf.py $(cat model_path.txt)")
        exit(0)
else:
    print("  설치된 GGUF 모델이 없습니다. 새로 다운로드합니다.")

print("\n다운로드 가능한 모델:")
for i, model in enumerate(models, 1):
    print(f"{i}. {model['name']}")
    print(f"   Repository: {model['repo_id']}")
    print(f"   File: {model['filename']}")
    print(f"   Size: {model.get('size', 'N/A')}")
    print()

choice = input(f"모델을 선택하세요 (1-{len(models)}) [기본값: 1]: ").strip() or "1"

try:
    idx = int(choice) - 1
    if idx < 0 or idx >= len(models):
        idx = 0
except:
    idx = 0

selected = models[idx]

print(f"\n선택된 모델: {selected['name']}")
print(f"다운로드 중... (처음에는 시간이 걸릴 수 있습니다)")
print()

try:
    model_path = hf_hub_download(
        repo_id=selected['repo_id'],
        filename=selected['filename'],
        local_dir_use_symlinks=False
    )

    print(f"\n✓ 다운로드 성공!")
    print(f"  모델 경로: {model_path}")

    # 경로 저장
    with open("model_path.txt", "w") as f:
        f.write(model_path)

    print(f"\n모델 경로가 'model_path.txt'에 저장되었습니다.")
    print(f"\n다음 명령어로 사용할 수 있습니다:")
    print(f"  python agent_gguf.py $(cat model_path.txt)")

except Exception as e:
    print(f"\n✗ 다운로드 실패: {e}")
    print(f"\n대안:")
    print(f"1. 다른 모델 선택")
    print(f"2. HuggingFace 웹사이트에서 수동 다운로드:")
    print(f"   https://huggingface.co/{selected['repo_id']}/tree/main")
    print(f"3. 브라우저에서 다운로드 후 경로 지정:")
    print(f"   python agent_gguf.py /path/to/downloaded/model.gguf")
