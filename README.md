# EmotionRAG Pipeline

This repository outlines the full pipeline for EmotionRAG, from data preprocessing to evaluation.

---

### 📦 Dataset Processing

#### 1. (Optional) Download the MIT License

[MIT License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)

#### 2. Preprocess Dataset

```bash
python process_dataset.py \
  --dataset_path /home/tako/km/EmotionRAG/dataset/mentalchat16k \
  --output_dir dataset/
```

---

### 🔍 Build Embeddings

#### Step 1: Download Required Models

- [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)  
- [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)

---

#### 💬 Emotion Embeddings

##### Corpus Embedding

```bash
cd embedding/emotion

python embed_corpus.py \
  --model_path /home/tako/km/EmotionRAG/model_download/roberta-base-go-emotions \
  --input_path /home/tako/km/HW/data/dataset/corpus_dataset_with_id.jsonl \
  --output_path embeddings/corpus_emotion_logits.jsonl
```

##### Query Embedding

```bash
cd embedding/emotion

python embed_query.py \
  --model_path /home/tako/km/EmotionRAG/model_download/roberta-base-go-emotions \
  --input_path /home/tako/km/HW/data/dataset/eval_dataset_with_id.jsonl \
  --output_path embeddings/query_emotion_logits.jsonl
```

---

#### 🧠 Semantic Embeddings

##### Corpus Embedding

```bash
cd embedding/semantic

python embed_corpus.py \
  --model_path /home/tako/km/FlashRAG/model_download/e5-base-v2 \
  --input_path /home/tako/km/HW/data/dataset/corpus_dataset_with_id.jsonl \
  --output_dir embeddings/
```

##### Query Embedding

```bash
cd embedding/semantic

python embed_query.py \
  --model_path /home/tako/km/FlashRAG/model_download/e5-base-v2 \
  --input_path /home/tako/km/HW/data/dataset/eval_dataset_with_id.jsonl \
  --output_dir embeddings/
```

---

### 🗂️ Create Index

```bash
cd index

python build_index.py \
  --emotion_path /home/tako/km/HW/embedding/emotion/embeddings/corpus_emotion_logits.jsonl \
  --semantic_path /home/tako/km/HW/embedding/semantic/embeddings/corpus_embeddings.npy \
  --output_dir corpus_index/
```

---

### 🔎 Retrieval

```bash
cd retrieval

python semantic_to_emotion_retriever.py \
  --semantic_index /home/tako/km/HW/index/corpus_index/semantic.index \
  --emotion_index /home/tako/km/HW/index/corpus_index/emotion.index \
  --query_semantic /home/tako/km/HW/embedding/semantic/embeddings/query_embeddings.npy \
  --query_emotion /home/tako/km/HW/embedding/emotion/embeddings/query_emotion_logits.jsonl \
  --query_data /home/tako/km/EmotionRAG/dataset/eval_dataset_with_id.jsonl \
  --corpus_data /home/tako/km/EmotionRAG/dataset/corpus_dataset_with_id.jsonl \
  --output_path retrieved/retrieved_top_3.jsonl
```

---

### 🧠 Generation

#### Download Your SLM:

- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)  
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

```bash
cd generation

python generator_vllm.py \
  --input_path /home/tako/km/HW/retrieval/retrieved/retrieved_top_3.jsonl \
  --output_path output/ \
  --model_path /home/tako/km/FlashRAG/model_download/phi4-mini-instruct
```

---

### 📊 Evaluation

```bash
cd evaluation

python evaluator.py \
  --gen /home/tako/km/HW/generation/output/our_generation_combined.jsonl \
  --ref /home/tako/km/EmotionRAG/dataset/eval_dataset_with_id.jsonl \
  --out result/evaluation_results.jsonl
```

---

### 📁 Directory Structure Example

```
EmotionRAG/
├── dataset/
├── embedding/
│   ├── emotion/
│   └── semantic/
├── index/
├── retrieval/
├── generation/
├── evaluation/
├── model_download/
└── output/
```
