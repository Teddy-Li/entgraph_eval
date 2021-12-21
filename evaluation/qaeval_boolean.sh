#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="QA_eval"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


eval_set=$1
version=$2
eval_method=$3
eg_name=$4
eg_suff=$5
eg_feat_idx=$6
device_name=$7
debug=$8


python -u qaeval_chinese_scratch.py --eval_set "$eval_set" --version "$version" --eval_mode boolean --eval_method "$eval_method" --eg_root ../gfiles --eg_name "$eg_name" --eg_suff "$eg_suff" --eg_feat_idx "$eg_feat_idx" --max_spansize 100 --backupAvg --device_name "$device_name" ${debug}