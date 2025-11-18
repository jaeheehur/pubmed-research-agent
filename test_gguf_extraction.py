"""
Test script to diagnose GGUF model loading issues
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("GGUF Model Loading Diagnostic")
print("=" * 80)

# Test 1: Check llama_cpp_python installation
print("\n[Test 1] Checking llama_cpp_python installation...")
try:
    import llama_cpp
    print(f"✓ llama_cpp_python version: {llama_cpp.__version__}")
except ImportError as e:
    print(f"✗ Failed to import llama_cpp: {e}")
    sys.exit(1)

# Test 2: Check for GGUF models
print("\n[Test 2] Scanning for installed GGUF models...")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if not cache_dir.exists():
    print(f"✗ HuggingFace cache directory not found: {cache_dir}")
    sys.exit(1)

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

if not gguf_models:
    print("✗ No GGUF models found in cache")
    sys.exit(1)

print(f"✓ Found {len(gguf_models)} GGUF model(s):")
for i, model in enumerate(gguf_models, 1):
    print(f"  [{i}] {model['filename']} ({model['size_gb']:.2f} GB)")
    print(f"      Path: {model['path']}")

# Test 3: Try loading the first model
print(f"\n[Test 3] Attempting to load model: {gguf_models[0]['filename']}...")
print("This is where segfaults typically occur...")

try:
    from llama_cpp import Llama

    print("Creating Llama instance with minimal settings...")
    llm = Llama(
        model_path=gguf_models[0]['path'],
        n_ctx=512,  # Small context window
        n_gpu_layers=0,  # CPU only first
        verbose=False,
        n_threads=2  # Fewer threads
    )
    print("✓ Model loaded successfully with CPU!")

    # Try a simple generation
    print("\n[Test 4] Testing simple text generation...")
    response = llm("Hello", max_tokens=5, echo=False)
    print(f"✓ Generation successful: {response['choices'][0]['text']}")

    # Now try with GPU
    print("\n[Test 5] Testing with Metal GPU (n_gpu_layers=1)...")
    del llm  # Free memory

    llm_gpu = Llama(
        model_path=gguf_models[0]['path'],
        n_ctx=512,
        n_gpu_layers=1,  # Use Metal
        verbose=False,
        n_threads=2
    )
    print("✓ Model loaded successfully with Metal GPU!")

    response = llm_gpu("Hello", max_tokens=5, echo=False)
    print(f"✓ Generation successful: {response['choices'][0]['text']}")

    print("\n" + "=" * 80)
    print("All tests passed! The GGUF model is working correctly.")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error during model loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
