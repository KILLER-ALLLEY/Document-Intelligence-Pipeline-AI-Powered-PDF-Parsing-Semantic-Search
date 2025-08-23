import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from paths import KEYWORDS_FILE, SAVE_PATH, KEYWORD_EMBEDDINGS_FILE

# ---------------- CONFIG / INPUTS ----------------
keywords_file = KEYWORDS_FILE
keyword_embeddings_file = KEYWORD_EMBEDDINGS_FILE
save_path = SAVE_PATH
base_threshold = 0.45
short_sentence_threshold = 0.7
model_name = "all-MiniLM-L6-v2"
device = "cpu"

# ---------------- HELPERS ----------------
def safe_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((vectors.shape[0], 384), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def cosine_similarity_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    return np.dot(safe_normalize(A), safe_normalize(B).T)

def filter_valid_embeddings(items):
    valid = []
    for item in items:
        emb = item.get("embedding", [])
        if isinstance(emb, list) and len(emb) == 384 and not all(x == 0 for x in emb):
            valid.append(item)
    return valid

# ---------------- MAIN FUNCTION ----------------
def run_semantic_search(sentences_embeddings):
    """
    sentences_embeddings: list of dicts like
        {'text':..., 'page_num':..., 'embedding':[...], 'bbox':[...]}
    """

    valid_sentences = filter_valid_embeddings(sentences_embeddings)
    if not valid_sentences:
        # ⚠️ Return structured error for frontend
        return {"status": "error", "message": "No valid sentence embeddings found (PDF may be scanned or empty)."}

    try:
        with open(keywords_file, "r", encoding="utf-8") as f:
            keywords_data = json.load(f)
    except FileNotFoundError:
        return {"status": "error", "message": f"Keywords file not found: {keywords_file}"}

    if os.path.exists(keyword_embeddings_file):
        with open(keyword_embeddings_file, "r", encoding="utf-8") as f:
            keyword_embeddings = json.load(f)
    else:
        model = SentenceTransformer(model_name, device=device)
        keyword_embeddings = []
        all_variants, variant_map = [], {}
        for kw in keywords_data.get("keywords", []):
            for variant in kw.get("variants", []):
                all_variants.append(variant)
                variant_map[variant] = kw["term"]
        if all_variants:
            vectors = model.encode(all_variants, normalize_embeddings=True, convert_to_numpy=True)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for variant, vec in zip(all_variants, vectors.tolist()):
                keyword_embeddings.append({"term": variant_map[variant], "variant": variant, "embedding": vec})
            os.makedirs(os.path.dirname(keyword_embeddings_file), exist_ok=True)
            with open(keyword_embeddings_file, "w", encoding="utf-8") as f:
                json.dump(keyword_embeddings, f, indent=2, ensure_ascii=False)

    valid_keywords = filter_valid_embeddings(keyword_embeddings)
    if not valid_keywords:
        return {"status": "error", "message": "No valid ESG keywords found. Please update keywords in Settings."}

    sent_array = np.array([s["embedding"] for s in valid_sentences], dtype=np.float32)
    kw_array = np.array([k["embedding"] for k in valid_keywords], dtype=np.float32)

    sim_matrix = cosine_similarity_matrix(sent_array, kw_array)

    results = []
    for i, sent in enumerate(valid_sentences):
        word_count = len(sent["text"].split())
        threshold = short_sentence_threshold if word_count < 7 else base_threshold
        matched_idx = np.where(sim_matrix[i] >= threshold)[0]
        if matched_idx.size > 0:
            matches = [{"keyword": valid_keywords[idx]["term"],
                        "variant": valid_keywords[idx]["variant"],
                        "similarity": float(sim_matrix[i, idx])} 
                        for idx in matched_idx]
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({
                "sentence": sent["text"],
                "page_num": sent["page_num"],
                "bbox": sent.get("bbox", [0,0,0,0]),
                "keywords": matches,
                "applied_threshold": threshold
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Semantic search complete! Found {len(results)} sentences with keyword matches.")
    print(f"💾 Results saved to {save_path}")
    return results
