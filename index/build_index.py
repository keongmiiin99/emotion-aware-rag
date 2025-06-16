import argparse
import faiss
import numpy as np
import json
import os
from pathlib import Path

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion_path", required=True, help="Path to emotion logits JSONL file")
    parser.add_argument("--semantic_path", required=True, help="Path to semantic embeddings .npy file")
    parser.add_argument("--output_dir", required=True, help="Directory to save FAISS indexes and ID file")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- 감정 벡터 로드 및 정규화 ---
    emotion_vectors, emotion_ids = [], []
    with open(args.emotion_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            emotion_vectors.append(np.array(obj["logits"], dtype=np.float32))
            emotion_ids.append(obj["id"])
    emotion_vectors = np.stack(emotion_vectors)
    emotion_vectors = normalize(emotion_vectors)

    # --- 의미 벡터 로드 (이미 정규화됨 가정) ---
    semantic_vectors = np.load(args.semantic_path).astype(np.float32)

    # --- 유효성 검사 ---
    assert len(emotion_vectors) == len(semantic_vectors), "❌ 임베딩 수 불일치!"

    # --- FAISS 인덱스 생성 및 저장 ---
    emotion_index = faiss.IndexFlatIP(emotion_vectors.shape[1])
    semantic_index = faiss.IndexFlatIP(semantic_vectors.shape[1])
    emotion_index.add(emotion_vectors)
    semantic_index.add(semantic_vectors)

    faiss.write_index(emotion_index, os.path.join(args.output_dir, "emotion.index"))
    faiss.write_index(semantic_index, os.path.join(args.output_dir, "semantic.index"))

    with open(os.path.join(args.output_dir, "emotion_ids.json"), "w", encoding="utf-8") as f:
        json.dump(emotion_ids, f, ensure_ascii=False, indent=2)

    print(f"✅ FAISS 인덱스 저장 완료: {args.output_dir}")

if __name__ == "__main__":
    main()
