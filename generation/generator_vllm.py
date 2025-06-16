import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pathlib import Path

def build_prompt(query, top_docs):
    context = "\n\n".join([f"[Document {i+1}]\n{doc['text']}" for i, doc in enumerate(top_docs)])
    return f"""<|user|>
You are a helpful assistant. Use the following documents to answer the user's question.

{context}

Question: {query}

<|assistant|>"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to retrieved JSONL file with top-3 documents per query")
    parser.add_argument("--output_path", required=True, help="Path to save generated answers")
    parser.add_argument("--model_path", required=True, help="Path to local vLLM model (e.g., llama3)")
    args = parser.parse_args()

    # 모델 설정
    llm = LLM(
        model=args.model_path,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4000,
        max_num_batched_tokens=8192
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256
    )

    # 입력 및 프롬프트 생성
    queries = []
    metadata = []

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = build_prompt(item["query_text"], item["top3_docs"])
            queries.append(prompt)
            metadata.append({
                "query_id": item["query_id"],
                "query": item["query_text"]
            })

    # 생성 실행
    outputs = llm.generate(queries, sampling_params)

    # 결과 저장
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for meta, output in zip(metadata, outputs):
            answer = output.outputs[0].text.strip()
            json.dump({
                "query_id": meta["query_id"],
                "query": meta["query"],
                "answer": answer
            }, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 생성 완료: {args.output_path}")

if __name__ == "__main__":
    main()
