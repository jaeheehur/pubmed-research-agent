"""
Test raw generation from BioMistral to see what it's actually producing
"""
import os
os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce noise

from llama_cpp import Llama
from pathlib import Path

# Find BioMistral model
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
biomistral_path = None

for model_dir in cache_dir.glob("models--*BioMistral*"):
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                for gguf_file in snapshot.glob("*.gguf"):
                    if 'Q4' in gguf_file.name:
                        biomistral_path = str(gguf_file)
                        break

if not biomistral_path:
    print("BioMistral model not found!")
    exit(1)

print("Loading BioMistral model...")
llm = Llama(
    model_path=biomistral_path,
    n_ctx=2048,
    n_gpu_layers=1,
    verbose=False,
    n_threads=2,
    n_batch=512
)

print("✓ Model loaded\n")

# Test with a simple instruction
prompt = """[INST] Extract medical entities from this text and return ONLY a JSON object.

Text: Patients with diabetes received metformin 1000mg. Common adverse events included nausea and headache.

JSON: [/INST]

"""

print("Prompt:")
print(prompt)
print("\n" + "=" * 80)
print("Raw model output:")
print("=" * 80)

response = llm(
    prompt,
    max_tokens=512,
    temperature=0.1,
    echo=False,
    stop=["</s>", "[INST]"],
    repeat_penalty=1.1
)

output = response['choices'][0]['text']
print(output)
print("=" * 80)
print(f"\nOutput length: {len(output)} chars")
print(f"Contains '{{': {'{' in output}")
print(f"Contains '}}': {'}' in output}")

# Try to extract JSON
if '{' in output:
    start = output.find('{')
    end = output.rfind('}') + 1
    if end > start:
        json_part = output[start:end]
        print(f"\nExtracted JSON part ({len(json_part)} chars):")
        print(json_part)
