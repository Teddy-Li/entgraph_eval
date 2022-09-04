#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="QA_eval"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


eval_set=$1
version=$2
eval_method=$3
t5_size=$4

# These flags below can include ``debug'', `no_ref_cache`'' and ``no_triple_cache''
flag1=$5
flag2=$6
flag3=$7

pwd
python -u qaeval_chinese_scripts.py --eval_set "$eval_set" --version "$version" \
--fpath_base ../../QAEval/nc_final_samples_%s_%s.json \
--wh_fpath_base ../../QAEval/nc_wh_final_samples_%s_%s.json \
--negi_fpath_base ../../QAEval/nc_negi_final_samples_%s_%s.json \
--sliced_triples_dir ../../QAEval/nc_time_slices/ --slicing_method disjoint --time_interval 3 \
--sliced_triples_base_fn nc_typed_triples_%s_%s.json --eval_mode boolean \
--eval_method "$eval_method" --eg_root ../gfiles --eg_name global_graphs --eg_suff _gsim.txt --eg_feat_idx 1 \
--max_spansize 64 --result_dir ../gfiles/qaeval_en_results_prd/%s_%s/ --backupAvg \
--min_graphsize 20480 --max_context_size 3200 --bert_dir ../../bert_checkpoints/ --mt5_dir t5-base \
--batch_size 16 --refs_cache_dir ./cache_dir_en/refs_%s.json --triples_cache_dir ./cache_dir_en/triples_%s.json \
--tfidf_path ../../DrQA/scripts/retriever/nc_doc_db-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz \
--articleIds_dict_path ../../DrQA/scripts/retriever/nc_articleIds_by_partition.json --num_refs_bert1 5 --lang en \
--all_preds_set_path ../../QAEval/nc_all_pred_set.json --threshold_samestr 1 --T5_size "$t5_size" \
${flag1} ${flag2} ${flag3}