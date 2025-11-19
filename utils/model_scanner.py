"""GGUF Model Scanner Utility"""

from pathlib import Path
from typing import List, Dict


def scan_installed_gguf_models() -> List[Dict]:
    """
    Scan installed GGUF models from HuggingFace cache
    
    Returns:
        List of dictionaries containing model information
    """
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
                display_name = gguf_file.name

                gguf_models.append({
                    "display_name": display_name,
                    "model_name": model_name,
                    "filename": gguf_file.name,
                    "path": str(gguf_file),
                    "size_gb": gguf_file.stat().st_size / (1024**3),
                })

    return gguf_models
