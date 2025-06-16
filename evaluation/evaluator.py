import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gen', required=True, help='Generated result JSONL path')
parser.add_argument('--ref', required=True, help='Evaluation dataset JSONL path')
parser.add_argument('--out', required=True, help='Output result JSON path')
args = parser.parse_args()

generated_path = args.gen
eval_path = args.ref
output_path = args.out

# 2. 데이터 로드
with open(generated_path, "r") as f:
    generated_data = [json.loads(line) for line in f]

with open(eval_path, "r") as f:
    eval_data = [json.loads(line) for line in f]

# 3. ID 정렬 및 예측/참조 리스트 생성
gen_dict = {item["query_id"].split("_")[-1]: item["answer"] for item in generated_data}
eval_dict = {item["id"].split("_")[-1]: item["output"] for item in eval_data}
common_ids = sorted(set(gen_dict.keys()) & set(eval_dict.keys()))
predictions = [gen_dict[i] for i in common_ids]
references = [eval_dict[i] for i in common_ids]

# 4. BLEU 계산
smoothie = SmoothingFunction().method4
bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for pred, ref in zip(predictions, references)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

# 5. ROUGE 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
avg_rouge = {
    'rouge1': sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores),
    'rouge2': sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores),
    'rougeL': sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores),
}

# 6. BERTScore 계산
P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=False)
avg_bert = {
    'precision': P.mean().item(),
    'recall': R.mean().item(),
    'f1': F1.mean().item()
}

# 7. 단어 오버랩 기반 Precision/Recall/F1 계산
def compute_token_overlap_metrics(predictions, references):
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        overlap = sum((pred_counter & ref_counter).values())
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'f1': sum(f1s) / len(f1s)
    }

token_overlap = compute_token_overlap_metrics(predictions, references)

# 8. 결과 통합
final_results = {
    "BLEU": avg_bleu,
    "ROUGE": avg_rouge,
    "BERTScore": avg_bert,
    "TokenOverlap": token_overlap
}

# 9. JSON 파일로 저장
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"📄 전체 평가 결과가 저장되었습니다: {output_path}")
