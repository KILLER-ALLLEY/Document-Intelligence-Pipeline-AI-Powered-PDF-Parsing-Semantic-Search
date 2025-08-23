import json
import os
from sentence_transformers import SentenceTransformer
from paths import SAVE_PATH_SENTENCES

# ------------------ LOAD MODEL ------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def process_batch(batch):
    """Process a batch of sentences and return embeddings in same format."""
    
    valid_items = []
    valid_texts = []
    
    for s in batch:
        text = s.get("text", "").strip() if s.get("text") else ""
        if text:  
            valid_items.append(s)
            valid_texts.append(text)
        else:
            
            valid_items.append(s)
            valid_texts.append("empty") 
    
   
    if not valid_texts:
        return [
            {
                "page_num": s.get("page_num", -1),
                "text": s.get("text", ""),
                "bbox": s.get("bbox", []),
                "embedding": [0.0] * 384 
            }
            for s in batch
        ]
    
    try:
       
        vectors = model.encode(
            valid_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
       
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        vectors_list = vectors.tolist()
        
        
        result_embeddings = []
        for i, (s, text) in enumerate(zip(valid_items, valid_texts)):
            if text == "empty" or not s.get("text", "").strip():
              
                embedding = [0.0] * vectors.shape[1] if len(vectors) > 0 else [0.0] * 384
            else:
                embedding = vectors_list[i]
            
            result_embeddings.append({
                "page_num": s.get("page_num", -1),
                "text": s.get("text", ""),
                "bbox": s.get("bbox", []),
                "embedding": embedding
            })
        
        return result_embeddings
        
    except Exception as e:
        print(f"⚠️ Error encoding batch: {e}")
   
        return [
            {
                "page_num": s.get("page_num", -1),
                "text": s.get("text", ""),
                "bbox": s.get("bbox", []),
                "embedding": [0.0] * 384 
            }
            for s in batch
        ]

def create_embeddings(sentences_data, save_path=SAVE_PATH_SENTENCES):
    """
    sentences_data: List of pages with sentences (output from pdf_service.extract_to_json)
    Returns: List of embeddings with metadata and saves to JSON.
    Handles empty texts, missing bboxes, or empty pages gracefully.
    """
   
    all_sentences = []
    for page in sentences_data or []:
        for sentence in page.get("sentences", []) or []:
           
            text = sentence.get("text", "")
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            all_sentences.append({
                "page_num": page.get("page_num", -1),
                "text": text,
                "bbox": sentence.get("bbox", [])
            })

    if not all_sentences:
        print("⚠️ No sentences found in input data.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    BATCH_SIZE = 100  
    
    print(f"📊 Processing {len(all_sentences)} sentences in batches of {BATCH_SIZE}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_embeddings = []

    # Process in batches
    for i in range(0, len(all_sentences), BATCH_SIZE):
        batch = all_sentences[i:i + BATCH_SIZE]
        try:
            print(f"🔄 Processing batch {i//BATCH_SIZE + 1}/{(len(all_sentences) + BATCH_SIZE - 1)//BATCH_SIZE}")
            batch_embeddings = process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"✅ Processed sentences {i + 1} to {min(i + len(batch), len(all_sentences))} / {len(all_sentences)}")
        except Exception as e:
            print(f"⚠️ Error in batch {i//BATCH_SIZE + 1}: {e}")
            
            fallback_embeddings = [
                {
                    "page_num": s.get("page_num", -1),
                    "text": s.get("text", ""),
                    "bbox": s.get("bbox", []),
                    "embedding": [0.0] * 384
                }
                for s in batch
            ]
            all_embeddings.extend(fallback_embeddings)
            continue

    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving file: {e}")

    print("✅ All sentence embeddings complete!")
    print(f"📊 Total embeddings created: {len(all_embeddings)}")
    return all_embeddings

