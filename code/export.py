#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import pickle
import logging
import codecs
import numpy as np

from models import Generator, load_external_embeddings

parser = argparse.ArgumentParser(description='Apply pre-trained post-spec mapping to new WEs')

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
assert os.path.isfile(params.in_file)

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
mapping = Generator(params)
checkpoint = torch.load(params.model)
mapping.load_state_dict(checkpoint['model'])
mapping.eval()
out_dico, out_emb = load_external_embeddings(params, params.in_file)
if params.cuda:
    out_emb.cuda()
    mapping.cuda()
if params.cuda:
    out_emb = out_emb.cuda() 
with torch.no_grad():
    mapped_emb = mapping(out_emb).data.cpu().numpy()

all_keys = out_dico.word2id.keys()
fhel = codecs.open(params.out_file, "w")
for key in all_keys:
   hv = mapped_emb[out_dico.index(key)]
   hv = hv / np.linalg.norm(hv)
   hv = map(str, list(hv))
   hv = " ".join([str(key)] + hv) + "\n"
   fhel.write(hv)

fhel.close()
