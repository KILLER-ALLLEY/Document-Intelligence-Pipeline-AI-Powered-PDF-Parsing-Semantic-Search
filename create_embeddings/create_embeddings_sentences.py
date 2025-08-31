import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from paths import SAVE_PATH_SENTENCES


import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
# from paths import SAVE_PATH_SENTENCES

# ------------------ LOAD MODEL ------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def create_embeddings(sentences_data, save_path=SAVE_PATH_SENTENCES):
    """
    sentences_data: List of pages with sentences (output from pdf_service.extract_to_json)
    Returns: List of embeddings with metadata and saves to JSON.
    Handles empty texts, missing bboxes, or empty pages gracefully.
    """
    # Pre-allocate and extract all data in one pass
    all_sentences = []
    valid_indices = []
    valid_texts = []
    
    for page in sentences_data or []:
        for sentence in page.get("sentences", []) or []:
            text = sentence.get("text", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            sentence_data = {
                "page_num": page.get("page_num", -1),
                "text": text,
                "bbox": sentence.get("bbox", [])
            }
            all_sentences.append(sentence_data)
            
            # Only add non-empty texts for embedding
            if text.strip():
                valid_indices.append(len(all_sentences) - 1)
                valid_texts.append(text)

    if not all_sentences:
        print("⚠️ No sentences found in input data.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    print(f"📊 Processing {len(all_sentences)} sentences ({len(valid_texts)} non-empty)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize all embeddings with zeros
    embedding_dim = 384
    all_embeddings_data = []
    
    for sentence in all_sentences:
        all_embeddings_data.append({
            "page_num": sentence["page_num"],
            "text": sentence["text"],
            "bbox": sentence["bbox"],
            "embedding": [0.0] * embedding_dim
        })

    # Process all valid texts at once if we have any
    if valid_texts:
        try:
            print("🔄 Encoding all valid texts at once...")
            
            # Encode all at once with optimized settings
            vectors = model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=512,  # Larger batch size for better efficiency
                convert_to_tensor=False
            )
            
            # Ensure correct shape
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            
            # Assign embeddings to valid indices
            for i, sentence_idx in enumerate(valid_indices):
                all_embeddings_data[sentence_idx]["embedding"] = vectors[i].tolist()
                
            print(f"✅ Successfully encoded {len(valid_texts)} sentences")
            
        except Exception as e:
            print(f"⚠️ Error during encoding: {e}")
            print("Falling back to batch processing...")
            
            # Fallback to batch processing if single encoding fails
            BATCH_SIZE = 1000  # Larger batches for fallback
            
            for i in range(0, len(valid_texts), BATCH_SIZE):
                batch_texts = valid_texts[i:i + BATCH_SIZE]
                batch_indices = valid_indices[i:i + BATCH_SIZE]
                
                try:
                    print(f"🔄 Processing fallback batch {i//BATCH_SIZE + 1}/{(len(valid_texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
                    
                    batch_vectors = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=256
                    )
                    
                    if len(batch_vectors.shape) == 1:
                        batch_vectors = batch_vectors.reshape(1, -1)
                    
                    for j, sentence_idx in enumerate(batch_indices):
                        all_embeddings_data[sentence_idx]["embedding"] = batch_vectors[j].tolist()
                        
                except Exception as batch_e:
                    print(f"⚠️ Error in fallback batch {i//BATCH_SIZE + 1}: {batch_e}")
                    # Keep zero embeddings for failed batches
                    continue

    # Save results
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving file: {e}")

    print("✅ All sentence embeddings complete!")
    print(f"📊 Total embeddings created: {len(all_embeddings_data)}")
    return all_embeddings_data