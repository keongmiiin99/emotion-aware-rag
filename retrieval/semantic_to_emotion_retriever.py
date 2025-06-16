import argparse
import json
import faiss
import numpy as np
from pathlib import Path

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)

def cosine_similarity_np(vec1, mat):
    vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return np.dot(mat, vec1)

def load_jsonl(path, text_key="text"):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[obj["id"]] = obj.get(text_key, "")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_index", required=True, help="Path to semantic FAISS index")
    parser.add_argument("--emotion_index", required=True, help="Path to emotion FAISS index")
    parser.add_argument("--query_semantic", required=True, help="Path to query semantic .npy file")
    parser.add_argument("--query_emotion", required=True, help="Path to query emotion logits .jsonl")
    parser.add_argument("--query_data", required=True, help="Path to eval_dataset_with_id.jsonl")
    parser.add_argument("--corpus_data", required=True, help="Path to corpus_dataset_with_id.jsonl")
    parser.add_argument("--output_path", required=True, help="Path to save retrieval results")
    args = parser.parse_args()

    # Load FAISS indexes
    semantic_index = faiss.read_index(args.semantic_index)
    emotion_index = faiss.read_index(args.emotion_index)

    # Load query embeddings
    query_semantic = np.load(args.query_semantic).astype(np.float32)
    faiss.normalize_L2(query_semantic)

    with open(args.query_emotion, 'r') as f:
        query_emotion_logits = [json.loads(line)["logits"] for line in f]
    query_emotion = np.array(query_emotion_logits, dtype=np.float32)
    query_emotion = normalize(query_emotion)

    # Load text data
    query_texts = load_jsonl(args.query_data, text_key="input")
    corpus_texts = load_jsonl(args.corpus_data, text_key="text")

    results = []
    for i, (q_sem, q_emo) in enumerate(zip(query_semantic, query_emotion)):
        q_sem = q_sem.reshape(1, -1)
        q_emo = q_emo.reshape(1, -1)

        # Step 1: 의미 top-6
        _, top6_indices = semantic_index.search(q_sem, k=6)

        # Step 2: 감정 유사도 top-3
        top6_emotion_vectors = np.vstack([
            emotion_index.reconstruct(int(idx)) for idx in top6_indices[0]
        ])
        top6_emotion_vectors = normalize(top6_emotion_vectors)

        emotion_sim = cosine_similarity_np(q_emo[0], top6_emotion_vectors)
        top3_indices = top6_indices[0][np.argsort(emotion_sim)[::-1][:3]]

        # Save result
        query_id = f"doc_{i}"
        query_text = query_texts.get(query_id, "")
        top3_docs = []

        for idx in top3_indices:
            doc_id = f"doc_{idx}"
            doc_text = corpus_texts.get(doc_id, "")
            top3_docs.append({"id": doc_id, "text": doc_text})

        results.append({
            "query_id": query_id,
            "query_text": query_text,
            "top3_docs": top3_docs
        })

    # Save to file
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 저장 완료: {args.output_path}")

if __name__ == "__main__":
    main()
