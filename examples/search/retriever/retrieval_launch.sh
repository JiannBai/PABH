#!/usr/bin/env bash
# Retrieval server for Search-R1 / HopQA training.
# Index and corpus assets live under SEARCHR1_ROOT.

save_path=${SEARCHR1_ROOT:-$HOME/data/searchR1}

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.clean.jsonl
retriever_name=e5
retriever_path=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Sanity check before mmap'ing 60 GB
for f in "$index_file" "$corpus_file"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: missing $f" >&2
        echo "Expected layout under $save_path:" >&2
        echo "  e5_Flat.index, wiki-18.clean.jsonl" >&2
        exit 1
    fi
done

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000
