import argparse
from datasets import load_from_disk
import random
import json
from pathlib import Path

# === ID 부여 함수 ===
def add_eval_ids(input_path, output_path):
    new_docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            new_docs.append({
                "id": f"eval_{i}",
                "input": doc.get("input", ""),
                "output": doc.get("output", "")
            })
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in new_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

def add_corpus_ids(input_path, output_path):
    new_docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            new_docs.append({
                "id": f"doc_{i}",
                "text": doc.get("output", "")
            })
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in new_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

# === 메인 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to HF dataset (load_from_disk)")
    parser.add_argument("--output_dir", default="dataset", help="Output directory to save processed files")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드 및 instruction 제거
    ds = load_from_disk(args.dataset_path).remove_columns(["instruction"])
    train_ds = ds["train"]

    # 2. 평가셋 샘플링 기준: output < 300 words
    def word_count(text):
        return len(text.split())
    
    filtered_indices = [i for i, ex in enumerate(train_ds) if word_count(ex['output']) < 300]
    eval_indices = random.sample(filtered_indices, 3000)
    all_indices = set(range(len(train_ds)))
    corpus_indices = list(all_indices - set(eval_indices))

    eval_ds = train_ds.select(eval_indices)
    corpus_ds = train_ds.select(corpus_indices).remove_columns(["input"])

    # 3. 추가 필터링: output < 500
    def under_500_words(example):
        return len(example["output"].split()) < 500
    corpus_ds = corpus_ds.filter(under_500_words)

    # 4. JSONL 저장
    eval_path = Path(args.output_dir) / "eval_dataset.jsonl"
    corpus_path = Path(args.output_dir) / "corpus_dataset.jsonl"
    eval_ds.to_json(str(eval_path), lines=True, orient="records", force_ascii=False)
    corpus_ds.to_json(str(corpus_path), lines=True, orient="records", force_ascii=False)

    # 5. ID 부여 및 저장
    add_eval_ids(eval_path, Path(args.output_dir) / "eval_dataset_with_id.jsonl")
    add_corpus_ids(corpus_path, Path(args.output_dir) / "corpus_dataset_with_id.jsonl")

    print("✅ 데이터셋 처리 및 저장 완료.")

if __name__ == "__main__":
    main()
