"""
Test script to verify agent initialization and model loading
"""
import os
import sys
from pathlib import Path

print("=" * 80)
print("Agent Initialization Test")
print("=" * 80)

# Set environment to show all logs
os.environ['LOG_LEVEL'] = 'DEBUG'

# Scan for GGUF models
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
gguf_models = []
for model_dir in cache_dir.glob("models--*"):
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        continue
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        for gguf_file in snapshot.glob("*.gguf"):
            gguf_models.append({
                "filename": gguf_file.name,
                "path": str(gguf_file),
                "size_gb": gguf_file.stat().st_size / (1024**3),
            })

if not gguf_models:
    print("No GGUF models found!")
    sys.exit(1)

# Use the smallest model for testing
gguf_models.sort(key=lambda x: x['size_gb'])
test_model = gguf_models[0]

print(f"\nUsing model: {test_model['filename']} ({test_model['size_gb']:.2f} GB)")
print(f"Path: {test_model['path']}")

# Test 1: Initialize agent with GGUF model
print("\n" + "=" * 80)
print("Test 1: Initialize PubMedResearchAgentGGUF with LLM")
print("=" * 80)

try:
    from agent_gguf import PubMedResearchAgentGGUF

    agent = PubMedResearchAgentGGUF(
        model_path=test_model['path'],
        use_llm=True,
        n_gpu_layers=1
    )

    print(f"✓ Agent initialized")
    print(f"  - use_llm: {agent.use_llm}")
    print(f"  - extractor type: {type(agent.extractor).__name__}")

    # Check if extractor has 'available' attribute
    if hasattr(agent.extractor, 'available'):
        print(f"  - extractor.available: {agent.extractor.available}")

    # Check if extractor has 'llm' attribute
    if hasattr(agent.extractor, 'llm'):
        print(f"  - extractor.llm: {agent.extractor.llm is not None}")

    # Test extraction
    print("\n" + "=" * 80)
    print("Test 2: Extract entities from sample text")
    print("=" * 80)

    sample_text = """
    A randomized controlled trial of 250 patients (mean age 65±10 years,
    both male and female) with type 2 diabetes treated with metformin 1000mg daily.
    The study observed adverse events including gastrointestinal disturbances,
    lactic acidosis (n=2), and headache. Glycemic control improved significantly.
    """

    entities = agent.extractor.extract(sample_text)

    print(f"✓ Extraction completed")
    print(f"  - Drugs: {len(entities.drugs)}")
    print(f"  - Adverse Events: {len(entities.adverse_events)}")
    print(f"  - Diseases: {len(entities.diseases)}")
    print(f"  - Demographics: {entities.demographics}")

    if entities.drugs:
        print("\n  Extracted drugs:")
        for drug in entities.drugs:
            print(f"    - {drug}")

    if entities.adverse_events:
        print("\n  Extracted adverse events:")
        for ae in entities.adverse_events:
            print(f"    - {ae}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
