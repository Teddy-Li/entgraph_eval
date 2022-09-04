#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="QA_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


eval_set=$1
version=$2
eval_method=$3
eg_name=$4
eg_suff=$5
eg_feat_idx=$6
debug=$8

pwd
python -u qaeval_chinese_scripts.py --eval_set "$eval_set" --version "$version" --eval_mode negi_masking --eval_method "$eval_method" --eg_root ../gfiles --eg_name "$eg_name" --eg_suff "$eg_suff" --eg_feat_idx "$eg_feat_idx" --max_spansize 100 --backupAvg --tfidf_path /home/s2063487/DrQA/scripts/retriever/clue_doc_db-tfidf-ngram=2-hash=16777216-tokenizer=spacy-chinese.npz --articleIds_dict_path /home/s2063487/DrQA/scripts/retriever/articleIds_by_partition.json --num_refs_bert1 5 ${debug}