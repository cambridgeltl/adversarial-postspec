#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import pickle
import logging

from models import build_model
from trainer_batch import Trainer

parser = argparse.ArgumentParser(description='Adversarial post-processing')

parser.add_argument("--params", type=str, required=True, help="Pickled file with experiment settings")
parser.add_argument("--model", type=str, required=True, help="File with trained parameters")
parser.add_argument("--in_file", type=str, required=True, help="File with embeddings to be post-specialized")
parser.add_argument("--out_file", type=str, required=True, help="File where post-specialized embeddings are saved")
parser.add_argument("--verbose", type=str, default="debug", help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")

params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert os.path.isfile(params.params)
assert os.path.isfile(params.model)
if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)

def initialize_exp(params):
    """
    Initialize experiment.
    """
    pkl_file = open(params.params, 'rb')
    oldparams = pickle.load(pkl_file)
    vars(params).update(vars(oldparams))

    # create logger
    logging.basicConfig(level=getattr(logging, params.verbose.upper()))
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))

initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)

# export embeddings to a text format
trainer.reload_best(mdl=params.model)
trainer.export(params)
