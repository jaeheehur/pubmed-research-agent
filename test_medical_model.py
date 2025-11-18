"""
Test medical GGUF models (BioMistral or JSL-MedLlama) for entity extraction
"""
import os
import sys
from pathlib import Path

print("=" * 80)
print("Medical Model Entity Extraction Test")
print("=" * 80)

# Set environment to show all logs
os.environ['LOG_LEVEL'] = 'INFO'

# Scan for medical GGUF models
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
medical_models = []

for model_dir in cache_dir.glob("models--*"):
    model_name = model_dir.name.replace("models--", "").replace("--", "/")

    # Only select medical models
    if not any(x in model_name.lower() for x in ['bio', 'med', 'clinical']):
        continue

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        continue

    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue

        for gguf_file in snapshot.glob("*.gguf"):
            size_gb = gguf_file.stat().st_size / (1024**3)

            # Prefer Q4 or Q6 quantization, skip IQ1 (too low quality)
            if 'IQ1' in gguf_file.name:
                continue

            medical_models.append({
                "model_name": model_name,
                "filename": gguf_file.name,
                "path": str(gguf_file),
                "size_gb": size_gb,
            })

if not medical_models:
    print("No medical GGUF models found!")
    print("\nAvailable models should include:")
    print("  - BioMistral")
    print("  - JSL-MedLlama")
    print("  - Clinical-specific models")
    sys.exit(1)

# Sort by size and prefer smaller models for testing
medical_models.sort(key=lambda x: x['size_gb'])

print(f"\nFound {len(medical_models)} medical model(s):")
for i, model in enumerate(medical_models[:3], 1):  # Show top 3
    print(f"  [{i}] {model['filename']} ({model['size_gb']:.2f} GB)")

# Use the first suitable model
test_model = medical_models[0]

print(f"\nUsing: {test_model['filename']}")
print(f"Path: {test_model['path']}")

# Initialize agent with medical model
print("\n" + "=" * 80)
print("Initializing Agent with Medical Model")
print("=" * 80)

try:
    from agent_gguf import PubMedResearchAgentGGUF

    agent = PubMedResearchAgentGGUF(
        model_path=test_model['path'],
        use_llm=True,
        n_gpu_layers=1  # Use Metal GPU
    )

    print(f"✓ Agent initialized")
    print(f"  - Extractor: {type(agent.extractor).__name__}")
    print(f"  - Model available: {agent.extractor.available}")

    # Test with a medical abstract
    print("\n" + "=" * 80)
    print("Test: Extract Entities from Medical Abstract")
    print("=" * 80)

    medical_abstract = """
    In this randomized, double-blind, placebo-controlled trial, we enrolled 450 patients
    (mean age 58.3 ± 12.4 years, 245 male and 205 female, predominantly Caucasian) with
    type 2 diabetes mellitus. Patients received either metformin 1000mg twice daily or
    placebo for 24 weeks. The primary outcomes were glycemic control and adverse events.

    Common adverse events included diarrhea (15.2%), nausea (8.7%), and abdominal pain (6.3%).
    Serious adverse events included lactic acidosis (n=3, severe), hypoglycemia requiring
    hospitalization (n=5, moderate), and acute kidney injury (n=2, severe). One death occurred
    due to cardiovascular complications unrelated to the study drug.

    Metformin significantly reduced HbA1c levels compared to placebo (mean difference -1.2%,
    95% CI -1.5 to -0.9, p<0.001). No significant differences in BMI were observed between
    groups (metformin: 28.4 ± 4.2 kg/m², placebo: 28.7 ± 4.5 kg/m²).
    """

    print("Extracting entities...")
    entities = agent.extractor.extract(medical_abstract)

    print(f"\n✓ Extraction completed")
    print(f"\n📊 Results:")
    print(f"  - Drugs: {len(entities.drugs)}")
    print(f"  - Adverse Events: {len(entities.adverse_events)}")
    print(f"  - Diseases: {len(entities.diseases)}")
    print(f"  - Sample Size: {entities.demographics.get('sample_size', 0)}")
    print(f"  - Age: {entities.demographics.get('age', 'Unknown')}")
    print(f"  - Gender: {entities.demographics.get('gender', 'Unknown')}")
    print(f"  - Race: {entities.demographics.get('race', 'Unknown')}")

    if entities.drugs:
        print(f"\n💊 Drugs extracted:")
        for drug in entities.drugs:
            print(f"    - {drug.get('name', 'Unknown')}")

    if entities.adverse_events:
        print(f"\n⚠️  Adverse Events extracted:")
        for ae in entities.adverse_events[:5]:  # Show first 5
            severity = ae.get('severity', 'unknown')
            print(f"    - {ae.get('event', 'Unknown')} [{severity}]")
        if len(entities.adverse_events) > 5:
            print(f"    ... and {len(entities.adverse_events) - 5} more")

    if entities.diseases:
        print(f"\n🏥 Diseases extracted:")
        for disease in entities.diseases:
            print(f"    - {disease}")

    # Check if extraction used LLM or rule-based
    if len(entities.drugs) > 0 or len(entities.adverse_events) > 3:
        print("\n" + "=" * 80)
        print("✅ SUCCESS: Medical model is working correctly!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: Extraction seems to be using rule-based fallback")
        print("The medical model may not be generating valid JSON")
        print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
