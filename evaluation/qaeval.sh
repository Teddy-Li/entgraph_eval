#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="EG_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL




python -u qaeval_chinese_scratch.py --eval_set dev --version 15_30_triple_doc_disjoint_40000_2_lexic_wordnet --eval_mode boolean --eval_method bert2 --eg_feat_idx 4 --max_spansize 100 --debug --backupAvg