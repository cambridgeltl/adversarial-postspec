# adversarial-postspec
Code and scripts for our adversarial post-specialisation model, as a follow-up of the NAACL-HLT 2018 submission by Vulić et al.

To run the regression training, you should run the following command:
`python maxmargin_ffn_bow2.py <output_file>`

(Don't ask why, but if you want to experiment with Glove vectors instead, call `maxmargin_ffn_glove300.py`)

The output (post-specialised SimLex and SimVerb words) will be then stored in the `./results/` directory.

To evaluate with SimLex-999 (or SimVerb-3500), you have to call the evaluation script in the `./evaluation/` directory:
`python simlex_evaluator.py simlexorig999.txt ../results/<output_file>`

All SGNS-BOW2 and Glove300 vectors (and necessary subsets of vectors) are available from my Google Drive at the following link:

If you have any question, please send it to me at: iv250@cam.ac.uk
