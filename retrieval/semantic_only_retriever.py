import argparse
import json
import faiss
import numpy as np
from pathlib import Path

def load_jsonl(path, text_key="text"):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[obj["id"]] = obj.get(text_key, "")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_index", required=True, help="Path to FAISS semantic index")
    parser.add_argument("--query_semantic", required=True, help="Path to query semantic .npy file")
    parser.add_argument("--query_data", required=True, help="Path to query JSONL file")
    parser.add_argument("--corpus_data", required=True, help="Path to corpus JSONL file")
    parser.add_argument("--output_path", required=True, help="Path to save top-3 retrieval result JSONL")
    args = parser.parse_args()

    # Load data
    query_semantic = np.load(args.query_semantic).astype(np.float32)
    faiss.normalize_L2(query_semantic)
    semantic_index = faiss.read_index(args.semantic_index)

    query_texts = load_jsonl(args.query_data, text_key="input")
    corpus_texts = load_jsonl(args.corpus_data, text_key="text")

    results = []
    for i, q_sem in enumerate(query_semantic):
        q_sem = q_sem.reshape(1, -1)
        _, top3_indices = semantic_index.search(q_sem, k=3)

        query_id = f"doc_{i}"
        query_text = query_texts.get(query_id, "")
        top3_docs = []

        for idx in top3_indices[0]:
            doc_id = f"doc_{idx}"
            doc_text = corpus_texts.get(doc_id, "")
            top3_docs.append({"id": doc_id, "text": doc_text})

        results.append({
            "query_id": query_id,
            "query_text": query_text,
            "top3_docs": top3_docs
        })

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 저장 완료 (의미 기반 검색): {args.output_path}")

if __name__ == "__main__":
    main()
