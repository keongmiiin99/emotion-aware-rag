import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to emotion classification model")
    parser.add_argument("--input_path", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", required=True, help="Path to save logits output JSONL file")
    parser.add_argument("--text_key", default="input", help="Which key in JSON to analyze (e.g., 'text' or 'input')")
    args = parser.parse_args()

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 출력 디렉토리 생성
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # 입력 파일 처리 및 로짓 추출
    with open(args.input_path, "r", encoding="utf-8") as infile, open(args.output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing documents"):
            data = json.loads(line)
            doc_id = data["id"]
            text = data.get(args.text_key) or " "

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.squeeze().cpu().tolist()

            json.dump({"id": doc_id, "logits": logits}, outfile)
            outfile.write("\n")

    print(f"✅ 감정 로짓 추출 완료: {args.output_path}")

if __name__ == "__main__":
    main()
