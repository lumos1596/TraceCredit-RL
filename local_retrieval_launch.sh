#!/bin/bash

echo "retriever use python:"
which python


index_file=directory-to-index/e5_Flat.index
corpus_file=directory-to-wikicorpus/wiki-18-corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=directory-to-model/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
