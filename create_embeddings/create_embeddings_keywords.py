# create_embeddings_keywords.py
import json
import os
from sentence_transformers import SentenceTransformer
from paths import KEYWORDS_FILE, OUTPUT_FILE

# ------------------ CONFIG ------------------
keywords_file = KEYWORDS_FILE
output_file = OUTPUT_FILE
BATCH_SIZE = 1000  # adjust if needed
DEVICE = "cpu"     # use "cuda" if GPU is available

# ------------------ LOAD MODEL ------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def process_batch(batch):
    """Process a batch of keyword variants and return embeddings in same format"""
    vectors = model.encode(
        batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()
    return vectors

def create_keyword_embeddings():
    # Load keywords
    with open(keywords_file, "r", encoding="utf-8") as f:
        keywords_data = json.load(f)

    
    all_variants = []
    variant_mapping = {} 
    for kw in keywords_data["keywords"]:
        for variant in kw["variants"]:
            all_variants.append(variant)
            variant_mapping[variant] = kw["term"]

   
    batches = [all_variants[i:i+BATCH_SIZE] for i in range(0, len(all_variants), BATCH_SIZE)]
    all_embeddings = []

    for i, batch in enumerate(batches):
        try:
            vectors = process_batch(batch)
            for variant, vec in zip(batch, vectors):
                all_embeddings.append({
                    "term": variant_mapping[variant],
                    "variant": variant,
                    "embedding": vec
                })
            print(f"Processed batch {i+1}/{len(batches)}")
        except Exception as e:
            print(f"Error in batch {i+1}: {e}")
            break

    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_embeddings)} keyword embeddings to {output_file}")
    return all_embeddings


create_keyword_embeddings()
