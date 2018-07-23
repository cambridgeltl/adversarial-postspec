#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import torch
import pickle
import logging
import time
import json

from collections import OrderedDict

from models import build_model
from trainer_batch import Trainer
from evaluator import Evaluator

parser = argparse.ArgumentParser(description='Adversarial post-processing')

parser.add_argument("--seen_file", type=str, default="../vectors/distrib.vectors", help="Seen vectors file")
parser.add_argument("--adjusted_file", type=str, default="../vectors/ar.vectors", help="Adjusted vectors file")
parser.add_argument("--unseen_file", type=str, default="../vectors/prefix.vectors", help="Unseen vectors file")
parser.add_argument("--out_dir", type=str, default="../results/", help="Where to store experiment logs and models")
parser.add_argument("--dataset_file", type=str, default="../vocab/simlexsimverb.words", help="File with list of words from datasets")

parser.add_argument("--seed", type=int, default=3, help="Initialization seed")
parser.add_argument("--verbose", type=str, default="debug", help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
# embs
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="renorm", help="Normalize embeddings before training")
# mapping
parser.add_argument("--noise", type=bool, default=False, help="Add gaussian noise to G layers and D input")
parser.add_argument("--gen_layers", type=int, default=2, help="Generator layers")
parser.add_argument("--gen_hid_dim", type=int, default=2048, help="Generator hidden layer dimensions")
parser.add_argument("--gen_dropout", type=float, default=0.5, help="Generator dropout")
parser.add_argument("--gen_input_dropout", type=float, default=0.2, help="Generator input dropout")
parser.add_argument("--gen_lambda", type=float, default=1, help="Generator loss feedback coefficient")
parser.add_argument("--sim_loss", type=str, default="max_margin", help="Similarity loss: mse or max_margin")
parser.add_argument("--sim_margin", type=float, default=1, help="Similarity margin (for max_margin losse)")
parser.add_argument("--sim_neg", type=int, default=25, help="Similarity negative examples (for max_margin loss)")
parser.add_argument("--sim_lambda", type=float, default=1, help="Similarity loss feedback coefficient")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.5, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--sim_optimizer", type=str, default="sgd,lr=0.1", help="Similarity optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert os.path.isfile(params.seen_file)
assert os.path.isfile(params.adjusted_file)
if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)

def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    pickle.dump(params, open(os.path.join(params.out_dir, 'params.pkl'), 'wb'))

    # create logger
    #logging.basicConfig(filename=os.path.join(params.out_dir, 'train.log'), level=getattr(logging, params.verbose.upper()))
    logging.basicConfig(level=getattr(logging, params.verbose.upper()))
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logging.info('The experiment will be stored in %s' % params.out_dir)


initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
evaluator = Evaluator(trainer)

"""
Learning loop for Adversarial Training
"""
logging.info('----> ADVERSARIAL TRAINING <----\n\n')

# training loop
for n_epoch in range(params.n_epochs):

    logging.info('Starting adversarial training epoch %i...' % n_epoch)
    tic = time.time()
    n_words_proc = 0
    stats = {'DIS_COSTS': [], 'GEN_COSTS' : [], 'SIM_COSTS' : []}

    for n_iter in range(0, params.epoch_size, params.batch_size):

        # discriminator training
        for _ in range(params.dis_steps):
            trainer.dis_step(stats)

        # mapping training (discriminator fooling)
        n_words_proc += trainer.mapping_step(stats, params)

	# similarity training
	trainer.sim_step(stats)

        # log stats
        if n_iter % 500 == 0:
            stats_str = [('DIS_COSTS', 'Discriminator loss'),
			('GEN_COSTS', 'Generator loss'),
			('SIM_COSTS', 'Similarity loss'),]
            stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                         for k, v in stats_str if len(stats[k]) > 0]
            stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
            logging.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

            # reset
            tic = time.time()
            n_words_proc = 0
            for k, _ in stats_str:
                del stats[k][:]

    # embeddings / discriminator evaluation
    to_log = OrderedDict({'n_epoch': n_epoch})
    evaluator.all_eval(to_log)
    evaluator.eval_dis(to_log)

    VALIDATION_METRIC = 'mean_cosine'

    # JSON log / save best model / end of epoch
    logging.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logging.info('End of epoch %i.\n\n' % n_epoch)

    # update the learning rate (stop if too small)
    trainer.update_lr(to_log, VALIDATION_METRIC)
    if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
        logging.info('Learning rate < 1e-6. BREAK.')
        break

# export embeddings to a text format
trainer.reload_best()
#trainer.export(params)
trainer.heldoutall(params)
