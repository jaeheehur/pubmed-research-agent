"""
Test actual PubMed search with GGUF model extraction
"""
import os
os.environ['LOG_LEVEL'] = 'INFO'

from agent_gguf import PubMedResearchAgentGGUF
from pathlib import Path

print("=" * 80)
print("Real PubMed Search + GGUF Extraction Test")
print("=" * 80)

# Find BioMistral model
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
biomistral_path = None

for model_dir in cache_dir.glob("models--*BioMistral*"):
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                for gguf_file in snapshot.glob("*Q4*.gguf"):
                    biomistral_path = str(gguf_file)
                    break

if not biomistral_path:
    print("BioMistral model not found!")
    exit(1)

print(f"\nUsing model: {Path(biomistral_path).name}")

# Initialize agent
print("\nInitializing agent...")
agent = PubMedResearchAgentGGUF(
    model_path=biomistral_path,
    use_llm=True,
    n_gpu_layers=1
)

print("✓ Agent initialized\n")

# Do a real search
query = "metformin adverse events diabetes"
print(f"Searching PubMed: '{query}'")
print("Getting 3 recent articles...\n")

results = agent.search_and_extract(
    query=query,
    max_results=3,
    extract_entities=True
)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total articles found: {results['total_articles']}")
print(f"Entities extracted from: {len(results.get('entities', []))} articles\n")

# Check each article
for i, entity_data in enumerate(results.get('entities', []), 1):
    entities = entity_data['entities']
    print(f"[{i}] {entity_data['title'][:70]}...")
    print(f"    PMID: {entity_data['pmid']}")
    print(f"    ├─ Drugs: {len(entities.get('drugs', []))}")
    print(f"    ├─ Adverse Events: {len(entities.get('adverse_events', []))}")
    print(f"    ├─ Diseases: {len(entities.get('diseases', []))}")
    print(f"    └─ Demographics: age={entities['demographics'].get('age', 'Unknown')}, "
          f"gender={entities['demographics'].get('gender', 'Unknown')}, "
          f"sample_size={entities['demographics'].get('sample_size', 0)}")

    # Show extracted drugs
    if entities.get('drugs'):
        print(f"       Drugs: {', '.join([d['name'] for d in entities['drugs'][:3]])}")
    if entities.get('adverse_events'):
        print(f"       AEs: {', '.join([ae['event'] for ae in entities['adverse_events'][:3]])}")
    print()

print("=" * 80)
print("Test completed successfully!")
print("=" * 80)
