# -*- coding: utf-8 -*-

from logging import getLogger
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        
    def dist_mean_cosine(self, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        self.mapping.eval()
	src_emb = self.mapping(self.src_emb.weight).data
	tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        assert set(self.src_dico.word2id.keys()) == set(self.tgt_dico.word2id.keys())
        
        indices = [[self.src_dico.index(k), self.tgt_dico.index(k)] for k in self.src_dico.word2id.keys()]
        indices = torch.LongTensor(list(indices))
	indices = indices.cuda() if self.params.cuda else indices
        #mean_cosine = F.cosine_similarity(src_emb[indices[:, 0]], tgt_emb[indices[:, 1]])
	mean_cosine = (src_emb[indices[:, 0]] * tgt_emb[indices[:, 1]]).sum(1).mean()
        logger.info("Mean cosine: %.5f" % mean_cosine)
        to_log['mean_cosine'] = mean_cosine

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.dist_mean_cosine(to_log)

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(self.mapping(emb))
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(emb)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred
