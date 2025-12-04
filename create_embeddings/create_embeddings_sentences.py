import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from paths import SAVE_PATH_SENTENCES

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model on CPU
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def create_embeddings(sentences_data, save_path=SAVE_PATH_SENTENCES):

    all_sentences = []
    valid_indices = []
    valid_texts = []

    for page in sentences_data or []:
        for sentence in page.get("sentences", []) or []:
            text = sentence.get("text", "")
            if not isinstance(text, str):
                text = str(text) if text else ""

            entry = {
                "page_num": page.get("page_num", -1),
                "text": text,
                "bbox": sentence.get("bbox", [])
            }

            idx = len(all_sentences)
            all_sentences.append(entry)

            if text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)

    if not all_sentences:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    print(f"📊 Processing {len(all_sentences)} sentences ({len(valid_texts)} non-empty)")

    embedding_dim = 384
    all_embeddings_data = [
        {
            "page_num": s["page_num"],
            "text": s["text"],
            "bbox": s["bbox"],
            "embedding": [0.0] * embedding_dim
        }
        for s in all_sentences
    ]

    if valid_texts:
        print("🔄 Encoding safely with tiny batches...")
        SAFE_BATCH = 8

        for start in range(0, len(valid_texts), SAFE_BATCH):
            end = start + SAFE_BATCH
            batch_texts = valid_texts[start:end]
            batch_indices = valid_indices[start:end]

            print(f"➡️ Batch {start} to {end} (size {len(batch_texts)})")

            vectors = model.encode(
                batch_texts,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)

            for i, idx in enumerate(batch_indices):
                all_embeddings_data[idx]["embedding"] = vectors[i].tolist()

        print(f"✅ Encoded {len(valid_texts)} sentences safely")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving file: {e}")

    return all_embeddings_data