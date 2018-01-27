#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import codecs
import torch
from torch import nn
import numpy as np
import logging
from dictionary import Dictionary

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

def load_external_embeddings(params, emb_path):
    """
    Reload pretrained embeddings from a text file.
    """
    
    word2id = {}
    vectors = []

    # load pretrained embeddings
    _emb_dim_file = params.emb_dim
    with codecs.open(emb_path) as f:
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                vect[0] = 0.01
            assert word not in word2id
            assert vect.shape == (_emb_dim_file,), i
            word2id[word] = len(word2id)
            vectors.append(vect[None])

    logging.info("Loaded %i pre-trained word embeddings" % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if params.cuda else embeddings
    assert embeddings.size() == (len(word2id), params.emb_dim), ((len(word2id), params.emb_dim, embeddings.size()))

    return dico, embeddings


def normalize_embeddings(emb, types):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            emb.sub_(emb.mean(1, keepdim=True).expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
 
def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_external_embeddings(params, params.seen_file)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    tgt_dico, _tgt_emb = load_external_embeddings(params, params.adjusted_file)
    params.tgt_dico = tgt_dico
    tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
    tgt_emb.weight.data.copy_(_tgt_emb)

    # mapping
    if params.map_type == "linear":
        mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        if getattr(params, 'map_id_init', True):
            mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    else:
        raise Exception('Unknown mapping type: "%s"' % params.map_type)

    # discriminator
    discriminator = Discriminator(params)

    # cuda
    if params.cuda:
        src_emb.cuda()
        tgt_emb.cuda()
        mapping.cuda()
        discriminator.cuda()

    # normalize embeddings
    normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator