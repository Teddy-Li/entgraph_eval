# entgraph_eval_chinese

This repository contains the evaluation script for Chinese entailment graphs and their baselines.
This is part of the [Chinese Entailment Graph Project](..). The code here instantiates two evaluations:
evaluation by entailment detection and evaluation by question answering (as described in Section 6 and Section 7 
in our paper). This repository is based on the earlier [entgraph_eval](https://github.com/mjhosseini/entgraph_eval) 
repository developed for the evaluation of English entailment graphs on the entailment detection task (Levy-Holt), but 
has been heavily revised and extended for the Chinese language, and for the new evaluation task by boolean question 
answering.

## Evaluation by Entailment Detection

1. Data Preperation: Put the constructed entailment graphs from [entGraph_Zh](../entGraph_Zh) under `./gfiles`; also put the Chinese
Levy-Holt dataset built from [LevyHolt_Chinese](../Levy_Holt_Chinese) under `./gfiles/dev_ent_chinese` and
`./gfiles/test_ent_chinese` respectively; download the prediction results of the English entailment graphs on Levy-Holt 
dataset from , put them under `./gfiles/results_en/` and `./gfiles/results_en++/` respectively; 
for doing back-translation ensemble, also download the English entailment graphs from 
[here](https://dl.dropboxusercontent.com/s/j7sgqhp8a27qgcf/gfiles.zip), and copy the back-translated Levy-Holt
dataset from [LevyHolt_Chinese](../Levy_Holt_Chinese) under `./gfiles/ent/`;

2. In the following steps we assume the working directory to be `./evaluation/`.
 Evaluate local graphs, example command is as follows:
`
python -u eval_chinese.py --gpath ../../entGraph_3/typedEntGrDir_Chinese2_2 --dev --sim_suffix _sim.txt 
--method global_scores_orig_dev_apooling_binc --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 4 
--write --exactType --no_lemma_baseline --no_constraints --outDir results/pr_rec_orig_dev_exhaust_22 
--eval_range orig_exhaust --avg_pooling
`;

3. Evaluate global graphs, example command is as follows:
`
python -u eval_chinese.py --gpath ../../entGraph_3/typedEntGrDir_Chinese2_2 --dev --sim_suffix _binc_2_1e-4_1e-2.txt 
--method global_scores_orig_dev_apooling_binc_G2_1e-4_1e-2 --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 
--featIdx 1 --write --exactType --backupAvg --no_lemma_baseline --no_constraints 
--outDir results/pr_rec_orig_dev_exhaust_22 --eval_range orig_exhaust --avg_pooling
`;

4. For the best dev set setting, do predictions on test set, example command is as follows:
`
python -u eval_chinese.py --gpath ../../entGraph_3/typedEntGrDir_Chinese2_2 --test --sim_suffix _binc_2_1e-4_1e-2.txt 
--method global_scores_orig_test_apooling_binc_G2_1e-4_1e-2 --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 
--featIdx 1 --write --exactType --backupAvg --no_lemma_baseline --no_constraints 
--outDir results/pr_rec_orig_test_exhaust_22 --eval_range orig_exhaust --avg_pooling
` (for global graphs);

5. Calculate ensemble scores between Chinese entailment graph and the English entailment graph as in 
[Hosseini et al. 2018](https://aclanthology.org/Q18-1048/), example command is as follows (for dev set):
`
python -u eval_merged.py --en_input ../gfiles/results_en/pr_rec/global_scores_Y.txt --zh_input 
../gfiles/results/pr_rec_orig_dev_exhaust_22/global_scores_orig_dev_apooling_binc_2_1e-4_1e-2.txt_Y.txt
`; 
ensemble scores will by default be saved in `ROOT/gfiles/results/pr_rev_merged_dev/`;

6. Calculate ensemble scores between Chinese entailment graph and the English entailment graph as in 
[Hosseini et al. 2021](https://aclanthology.org/2021.findings-emnlp.238/), example command is as follows (for dev set):
`
python -u eval_merged.py --en_input ../gfiles/results_en++/Aug_context_MC_dev_global_Y.txt 
--zh_input ../gfiles/results/pr_rec_orig_dev_exhaust_22/global_scores_orig_dev_apooling_binc_2_1e-4_1e-2.txt_Y.txt
--output ../gfiles/results/pr_rec_merged_dev_plusplus/scores_%s.txt
--output_Y ../gfiles/results/pr_rec_merged_dev_plusplus/scores_Y_%s.txt
`; ensemble scores will then be saved in `ROOT/gfiles/results/pr_rec_merged_dev_plusplus/`;

7. Plot out the precision-recall curves using `ROOT/gfiles/plotting_merged.py`, example command is as follows (for dev set):
`
python plotting_merged.py --mode baselines_dev
`; ⚠️ remember to change the file paths in `input_list`, especially, set the `\gamma` values to the best dev set results
from your experiment!

8. In order to evaluate the back-translated ensemble in our ablation study 3 in Section 6, get prediction scores of the 
English entailment graph on the back-translated Levy-Holt dataset using:
`
python eval.py --gpath global_graphs --dev_bt --sim_suffix _gsim.txt --method global_scores --CCG 1 --typed 1 
--supervised 0 --oneFeat 1 --useSims 0 --featIdx 1 --exactType --backupAvg --write --outDir results_en/pr_rec_bt
`;

9. Calculate back-translation ensemble scores (dev set) using:
`
python eval_merged.py --en_input ../gfiles/results_en/pr_rec/global_scores_Y.txt --zh_input 
../gfiles/results_en/pr_rec_bt/global_scores_Y.txt --output ../gfiles/results_en/pr_rec_merged_dev_bt/scores_%s.txt 
--output_Y ../gfiles/results_en/pr_rec_merged_dev_bt/scores_Y_%s.txt
` (test set command is similar);

For all steps, please reset the file paths according to your need 😊;


## Evaluation by Question Answering

1. Prepare QA Eval dataset from [QAEval](../QAEval), specifically, keep at hand `QAEval/clue_time_slices/*` and
`QAEval/clue_final_samples_15_30_triple_doc_disjoint_40000_2_lexic_wordnet_[dev/test].json`;

2. Run QA Evaluation, example commands are as follows:
    - for BERT_tfidf baseline: `sh qaeval_boolean.sh [dev/test] 15_30_triple_doc_disjoint_40000_2_lexic_wordnet bert1 - - 4 "cuda:0"`
    - for BERT_sent baseline: `sh qaeval_boolean.sh [dev/test] 15_30_triple_doc_disjoint_40000_2_lexic_wordnet bert2 - - 4 "cuda:0"`
    - for BERT_rel baseline: `sh qaeval_boolean.sh [dev/test] 15_30_triple_doc_disjoint_40000_2_lexic_wordnet bert3 - - 4 "cuda:0"`
    - for DDPORE baseline: `sh qaeval_boolean.sh [dev/test] 15_30_triple_doc_disjoint_40000_2_lexic_wordnet eg 
    ../../entGraph_Zh/typedEntGrDir_Chinese_BSL2_2 _binc_1_1e-4_1e-2.txt 1 "cpu"`
    - for EG_Zh (ours): `sh qaeval_boolean.sh [dev/test] 15_30_triple_doc_disjoint_40000_2_lexic_wordnet eg 
    ../../entGraph_Zh/typedEntGrDir_Chinese2_2 _binc_2_1e-4_1e-2.txt 1 "cpu"`
    
    The results should be displayed on command line output, and stored under `ROOT/gfiles/qaeval_results/`;

3. Plot out the precision-recall curves using `ROOT/gfiles/plotting_merged.py` example command is as follows:
`
python plotting_merged.py --mode dev_qaeval
`; remember to change the file paths according to your need! 🎉

## Cite Us

Coming soon.