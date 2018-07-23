Code for the paper:

"Adversarial Propagation and Zero-Shot Cross-Lingual Transfer of Word Vector Specialization"
Edoardo Maria Ponti, Ivan Vulić, Goran Glavaš, Nikola Mrkšić, and Anna Korhonen 
Submitted to EMNLP 2018

Directory structure:

code: contains main scripts to train and test AuxGAN for post-specialization
evaluation: script and databases for intrinsic evaluation
results: where the output of the post-specialization procedure is saved
vectors: contains the training data (i.e. original vectors and vectors specialized by Attract-Repel)
vocab: list of words to be excluded from the training data because they are present in the evaluation databases.

How to run:

> cd code
> python adversarial.py --seen_file ../vectors/SEEN_VECTORS --adjusted_file ../vectors/AR_SPECIALIZED_VECTORS \\
>     --unseen_file ../vectors/ALL_ORIGINAL_VECTORS --out_dir ../results/EXPERIMENT_NAME 

For additional options, inspect the file "adversarial.py"
After completing the epochs, the script saves two files in the folder specified as "out_dir": gold_embs.txt and silver_embs.txt. They are based on two different settings where AR specialized vectors are saved when available (gold) and only post-specialized vectors are saved (silver). The paper reports the gold setting.

Code and scripts for our adversarial post-specialisation model, as a follow-up of the NAACL-HLT 2018 submission by Vulić et al.

To run the regression training, you should run the following command: python maxmargin_ffn_bow2.py <output_file>

(Don't ask why, but if you want to experiment with Glove vectors instead, call maxmargin_ffn_glove300.py)

The output (post-specialised SimLex and SimVerb words) will be then stored in the ./results/ directory.

To evaluate with SimLex-999 (or SimVerb-3500), you have to call the evaluation script in the ./evaluation/ directory: python simlex_evaluator.py simlexorig999.txt ../results/<output_file>

All SGNS-BOW2 and Glove300 vectors (and necessary subsets of vectors) are available from my Google Drive at the following link: https://drive.google.com/open?id=1K5VJTHFPql5WvtYB-GiZrgSgayh54KNB

(You need to place the extracted vectors under sub-directory ./vectors)

If you have any question, please send it to me at: iv250@cam.ac.uk
