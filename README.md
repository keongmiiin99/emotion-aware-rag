# Emotion-aware RAG Pipeline

This repository outlines the full pipeline for Emotion-aware RAG, from data preprocessing to evaluation.
> ⚠️ **Important:** All file and folder paths in the commands below are examples.  
> Please replace them with your own file and directory paths.

---
### 🛠️ Setup: Create Virtual Environment & Install Dependencies

```bash
# Create a virtual environment and  Install required dependencies
pip install -r requirements.txt
```

---

### 📦 Dataset Processing

#### Step 1: Download Required Dataset

- [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)


#### Step 2 : Preprocess Dataset

```bash
cd data
python process_dataset.py \
  --dataset_path /home/~/mentalchat16k \
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
  --model_path /home/~/roberta-base-go-emotions \
  --input_path /home/~/data/dataset/corpus_dataset_with_id.jsonl \
  --output_path embeddings/corpus_emotion_logits.jsonl
```

##### Query Embedding

```bash
cd embedding/emotion

python embed_query.py \
  --model_path /home/~/roberta-base-go-emotions \
  --input_path /home/~/data/dataset/eval_dataset_with_id.jsonl \
  --output_path embeddings/query_emotion_logits.jsonl
```

---

#### 🧠 Semantic Embeddings

##### Corpus Embedding

```bash
cd embedding/semantic

python embed_corpus.py \
  --model_path /home/~/e5-base-v2 \
  --input_path /home/~/data/dataset/corpus_dataset_with_id.jsonl \
  --output_dir embeddings/
```

##### Query Embedding

```bash
cd embedding/semantic

python embed_query.py \
  --model_path /home/~/e5-base-v2 \
  --input_path /home/~/data/dataset/eval_dataset_with_id.jsonl \
  --output_dir embeddings/
```

---

### 🗂️ Create Index

```bash
cd index

python build_index.py \
  --emotion_path /home/~/embedding/emotion/embeddings/corpus_emotion_logits.jsonl \
  --semantic_path /home/~/embedding/semantic/embeddings/corpus_embeddings.npy \
  --output_dir corpus_index/
```

---

### 🔎 Retrieval

```bash
cd retrieval

python semantic_to_emotion_retriever.py \
  --semantic_index /home/~/index/corpus_index/semantic.index \
  --emotion_index /home/~/index/corpus_index/emotion.index \
  --query_semantic /home/~/embedding/semantic/embeddings/query_embeddings.npy \
  --query_emotion /home/~/embedding/emotion/embeddings/query_emotion_logits.jsonl \
  --query_data /home/~/dataset/eval_dataset_with_id.jsonl \
  --corpus_data /home/~/dataset/corpus_dataset_with_id.jsonl \
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
  --input_path /home/~/retrieval/retrieved/retrieved_top_3.jsonl \
  --output_path output/ \
  --model_path /home/~/phi4-mini-instruct
```

---

### 📊 Evaluation

```bash
cd evaluation

python evaluator.py \
  --gen /home/~/generation/output/our_generation_combined.jsonl \
  --ref /home/~/dataset/eval_dataset_with_id.jsonl \
  --out result/evaluation_results.jsonl
```

---

### 📁 Directory Structure Example

```
Emotion-aware-RAG/
├── dataset/
├── embedding/
│   ├── emotion/
│   └── semantic/
├── index/
├── retrieval/
├── generation/
└── evaluation/
```
