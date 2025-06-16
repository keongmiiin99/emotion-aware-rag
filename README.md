# emotion-aware-rag

### process dataset

Please download the dataset from here :  https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md

```
python process_dataset.py \
--dataset_path /home/tako/km/EmotionRAG/dataset/mentalchat16k \
--output_dir dataset/
```

### Build embeddings

First download these models

- https://huggingface.co/intfloat/e5-base-v2
- https://huggingface.co/SamLowe/roberta-base-go_emotions

Emotion embeddings

`corpus`

cd embedding/emotion

python embed_corpus.py   --model_path /home/tako/km/EmotionRAG/model_do
wnload/roberta-base-go-emotions   --input_path /home/tako/km/HW/data/dataset/corpus_dataset_with_id.jsonl   --output_path
embeddings/corpus_emotion_logits.jsonl

`query` 

cd embedding/emotion

python embed_query.py \
--model_path /home/tako/km/EmotionRAG/model_download/roberta-base-go-emotions \
--input_path /home/tako/km/HW/data/dataset/eval_dataset_with_id.jsonl \
--output_path embeddings/query_emotion_logits.jsonl

Semantic embeddings

`corpus`

cd embedding/semantic

python embed_corpus.py \
--model_path /home/tako/km/FlashRAG/model_download/e5-base-v2 \
--input_path /home/tako/km/HW/data/dataset/corpus_dataset_with_id.jsonl \
--output_dir embeddings/

`query`

cd embedding/semantic

python embed_query.py   --model_path /home/tako/km/FlashRAG/model_down
load/e5-base-v2   --input_path /home/tako/km/HW/data/dataset/eval_dataset_with_id.jsonl   --output_dir embeddings/

### create index

cd index

python build_index.py \

--emotion_path /home/tako/km/HW/embedding/emotion/embeddings/corpus_emotion_logits.jsonl \

--semantic_path /home/tako/km/HW/embedding/semantic/embeddings/corpus_embeddings.npy \

--output_dir corpus_index/

### Retrieval

cd retrieval

python semantic_to_emotion_retriever.py --semantic_index /home/tako/km/HW/index/corpus_index/semantic.index --emotion_index /home/tako/km/HW/index/corpus_index/emotion.index --query_semantic /home/tako/km/HW/embedding/semantic/embeddings/query_embeddings.npy --query_emotion /home/tako/km/HW/embedding/emotion/embeddings/query_emotion_logits.jsonl --query_data /home/tako/km/EmotionRAG/dataset/eval_dataset_with_id.jsonl --corpus_data /home/t
ako/km/EmotionRAG/dataset/corpus_dataset_with_id.jsonl --output_path retrieved/retrieved_top_3.jsonl

### generate

First download your slm 

we used these models

- https://huggingface.co/microsoft/Phi-4-mini-instruct
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

cd generation

python generator_vllm.py --input_path /home/tako/km/HW/retrieval/retrieved/retrieved_top_3.jsonl --output_path output/ --model_path /home/tako/km/FlashRAG/model_download/phi4-mini-instruct

### evaluate

cd evaluation

python [evaluator.py](http://evaluator.py/) --gen /home/tako/km/HW/generation/output/our_generation_combined.jsonl --ref /home/tako/km/EmotionRAG/dataset/eval_dataset_with_id.jsonl --out result/evaluation_results.jsonl
