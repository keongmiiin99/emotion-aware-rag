import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to SentenceTransformer model")
    parser.add_argument("--input_path", required=True, help="Path to input JSONL file with 'id' and 'text'")
    parser.add_argument("--output_dir", default="outputs/embeddings", help="Directory to save embeddings and metadata")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_path)

    ids, texts = [], []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["id"])
            texts.append(doc["text"])

    embeddings = model.encode(
        ["passage: " + t for t in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "corpus_embeddings.npy", embeddings)
    with open(output_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    with open(output_dir / "texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {output_dir}")

if __name__ == "__main__":
    main()

