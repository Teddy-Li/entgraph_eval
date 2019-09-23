This is the evaluation code on the entailment datasets for the following paper:

Learning Typed Entailment Graphs with Global Soft Constraints, Mohammad Javad Hosseini, Nathanael Chambers, Siva Reddy, Xavier Holt, Shay B. Cohen, Mark Johnson, and Mark Steedman. Transactions of the Association for Computational Linguistics (TACL 2018).

Duality of Link Prediction and Entailment Graph Induction,  Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Association for Computational Linguistics (ACL 2019).



### How to Run the Code

Please follow the below instructions to create entailment graphs and/or replicate the paper's experiments.

**Step 1**: Clone the entgraph_eval project and download necessary files.

    git clone https://github.com/mjhosseini/entgraph_eval.git
    wget https://dl.dropboxusercontent.com/s/j7sgqhp8a27qgcf/gfiles.zip
    unzip gfiles.zip
    rm gfiles.zip

**Step 2**: Add the learned entailment graphs folder inside the gfiles folder. You can also download and unzip learned global_graphs from https://worksheets.codalab.org/worksheets/0x8684ad8e95e24c4d80074278bce37ba4.

**Step 3**: Install dependencies (if required)

    pip install numpy
    pip install scipy
    pip install nltk
    python -m nltk.downloader wordnet
    python -m nltk.downloader verbnet
    python -m nltk.downloader stopwords
    pip install sklearn

**Step 4**: Run the evaluation script.

    cd entgraph_eval/evaluation/
    python eval.py --gpath global_graphs --dev --sim_suffix _gsim.txt --method global_scores --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 1 --exactType --backupAvg --write
    
The code writes a file called pr_ The code writes the precisions and recalls of a few baselines. It also writes the precision and recall for the method at precision~0.75.

The main parameters are these ones:

--sim_suffix: The suffix of the entailment graph files, e.g., _gsim.txt.

featIdx=1: If there are more than one similarity measures in the entailment graph files (e.g., local similarity and global similarity in the global_scores folder), this index specifies which similarity measure should be used.

--exactType --backupAvg: If you add these two options together, the code first tries to use the similarity measure of the graph with the same types as the entailment query. For example, for (PERSON visit LOCATION) => (PERSON arrives in LOCATION), it will use the similarity measure for the (PERSON,LOCATION) graph. If that graph doesn't have the relations of interest (visit or arrives in), then the code looks at the uniform average of the scores for those relations across all the graphs. If only --exactType is used, then the similarity will be 0 if the graph doesn't have the relations of interest. Finally, if none of these options are used, the code always uses the uniform avarage of the similarity scores across all graphs.

--method: A given name to the similarity measures (e.g., global_scores in our case).

Other parameters that should be mainly not changed:

CCG=1: Evaluate based on entailment graphs and datasets with CCG parser extractions (0 means openIE style).

typed=1: Use the types of arguments. If set to 0, it will ignore all the types, but the entailment graph should also be untyped (e.g., only thing#thing_sim.txt as the only entailment graph).

supervied=0: All the experiments are unsupervised

useSims=0: If we set this to 1, the code will look into other relations that have high similarity measures to the relations of interest in GloVe embedding space. We did not use this option in the above papers, but it will improve the results slightly. If you're interested in testing other embedding spaces, you can do that by providing a file in the format of gfiles/ent/ccg.sim.

oneFeat=1: This means that we only use one of the similarity measures and don't combine them in any way.

--no_lemma_baseline: It won't run the lemma_baseline in advance. All the results in the papers are WITH the lemma baseline.

--no_constraints: It won't use the extra constraints in advance. All the results in the papers are WITH the additional constraints.

**Step 5**: (Optional) The above code might take a few hours to run. The reason is that for the provided entailment datasets, it will look into all the entailment graphs to extract the relevant similarity features. If you run the code once on some entailment graphs and are interested in changing some of the options (e.g., the featIdx), you don't need to run the whole code again. You can simply run the script below, which only uses the extracted similarity measures for those entailment datasets. This code just takes a few minutes to complete. In that case, it's suggested to run step 4 with useSims=1, because this option can't be turned on in this step otherwise.

    cp feats/feats_global_graphs.txt 
    python eval.py --featsFile feats_global_graphs --dev --method global_scores --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 1 --exactType --backupAvg --write

--featsFile: 
