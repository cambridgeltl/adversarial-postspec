import os, re
import logging
import numpy as np
import inspect
import codecs

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from models import load_external_embeddings, normalize_embeddings

def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params

class MaxMargin_Loss(torch.nn.Module):

    def __init__(self, params):
        super(MaxMargin_Loss,self).__init__()
        self.params = params

    def forward(self, y_pred, y_true):
	cost = 0.
	for i in xrange(0, self.params.sim_neg):
            new_true = torch.randperm(self.params.batch_size)
            new_true = new_true.cuda() if self.params.cuda else new_true
            new_true = y_true[new_true]
            mg = self.params.sim_margin - F.cosine_similarity(y_true, y_pred) + F.cosine_similarity(new_true, y_pred)
	    cost += torch.clamp(mg, min=0)
        return cost.mean()

class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = params.tgt_dico
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
	# optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        if hasattr(params, 'sim_optimizer'):
            optim_fn, optim_params = get_optimizer(params.sim_optimizer)
            self.sim_optimizer = optim_fn(mapping.parameters(), **optim_params)
	self.max_margin = MaxMargin_Loss(params)
        # best validation score
        self.best_valid_metric = 0

        self.decrease_lr = False

    def get_sim_xy(self):
        """
        Get similarity input batch / output target.
        """
	# select random word IDs
        bs = self.params.batch_size
     	src_ids = np.random.randint(0, len(self.src_dico), bs)
    	words = [self.src_dico[si] for si in src_ids]
        tgt_ids = [self.tgt_dico.index(w) for w in words]
        src_ids = torch.LongTensor(src_ids)
    	tgt_ids = torch.LongTensor(tgt_ids)
    	if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
	tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=False))
        tgt_emb = Variable(tgt_emb.data, volatile=False)

        return src_emb, tgt_emb

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico))
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico))
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = self.src_emb(Variable(src_ids, volatile=True))
        tgt_emb = self.tgt_emb(Variable(tgt_ids, volatile=True))
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        yp = torch.FloatTensor([self.params.dis_smooth] * bs)
	yn = torch.FloatTensor([1 - self.params.dis_smooth] * bs)
        yp = Variable(yp.cuda() if self.params.cuda else yp)
	yn = Variable(yn.cuda() if self.params.cuda else yn)

        return src_emb, tgt_emb, yp, yn

    def sim_step(self, stats):
	"""
	Train the similarity between mapped src and tgt
	"""
	self.discriminator.eval()
	# loss
        x, y = self.get_sim_xy()
        ycos = torch.Tensor([1.] * self.params.batch_size)
        ycos = ycos.cuda() if self.params.cuda else ycos
        if self.params.sim_loss == "mse":
            loss = F.cosine_embedding_loss(x, y, Variable(ycos))
        elif self.params.sim_loss == "max_margin":
            loss = self.max_margin(x, y)
        else:
            raise Exception('Unknown similarity loss: "%s"' % self.params.sim_loss)
	loss = self.params.sim_lambda * loss
    	stats['SIM_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logging.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.sim_optimizer.zero_grad()
        loss.backward()
        self.sim_optimizer.step()

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        xp, xn, yp, yn = self.get_dis_xy(volatile=True)
	for x, y in [(xp, yp), (xn, yn)]:
            preds = self.discriminator(Variable(x.data))
            loss = self.params.dis_lambda * F.binary_cross_entropy(preds, y)
	    stats['DIS_COSTS'].append(loss.data[0])

            # check NaN
            if (loss != loss).data.any():
                logging.error("NaN detected (discriminator)")
                exit()

            # optim
            self.dis_optimizer.zero_grad()
            loss.backward()
            self.dis_optimizer.step()
            clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats, params):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        xp, xn, yp, yn = self.get_dis_xy(volatile=False)
	x = torch.cat([xp, xn], 0)
	y = torch.cat([yp, yn], 0)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
	loss = self.params.gen_lambda * loss
        stats['GEN_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logging.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

        return 2 * self.params.batch_size

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logging.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1:
            if to_log[metric] < self.best_valid_metric:
                logging.info("Validation metric is lower than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logging.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logging.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            path = os.path.join(self.params.out_dir, 'best_mapping.t7')
            checkpoint = {'model': self.mapping.state_dict()}
	    logging.info('* Saving the mapping parameters to %s ...' % path)
            torch.save(checkpoint, path)

    def reload_best(self, mdl="default"):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.out_dir, 'best_mapping.t7') if mdl == "default" else mdl
        logging.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        checkpoint = torch.load(path)
        self.mapping.load_state_dict(checkpoint['model'])

    def heldoutall(self, params):
        logging.info("Exporting mapped embeddings...")
        self.mapping.eval()
        out_dico, out_emb = load_external_embeddings(params, params.unseen_file)
        params.out_dico = out_dico
        mapped_emb = self.mapping(Variable(out_emb, volatile=True)).data.cpu().numpy()
        ar_emb = self.tgt_emb.weight.data.cpu().numpy() 
        logging.info("Reading SimLex and SimVerb words...")

        # Now translate the unseen words to the target AR-specialised vector space
        all_keys = out_dico.word2id.keys()
        fhel = codecs.open(params.out_dir + "gold_embs.txt", "w")
        fall = codecs.open(params.out_dir + "silver_embs.txt", "w")
        for key in all_keys:
            hv = mapped_emb[out_dico.index(key)]
            hv = hv / np.linalg.norm(hv)
            hv = map(str, list(hv))
            hv = " ".join([str(key)] + hv) + "\n"
            fhel.write(hv)
            if key in self.tgt_dico.word2id:
                av = ar_emb[self.tgt_dico.index(key)]
                av = av / np.linalg.norm(av)
                av = map(str, list(av))
                av = " ".join([str(key)] + av) + "\n"
                fall.write(av)
            else:
                fall.write(hv)

        fhel.close()
        fall.close()
        logging.info("...Done!")

    def export(self, params):
        logging.info("Exporting mapped embeddings...")
        self.mapping.eval()
        out_dico, out_emb = load_external_embeddings(params, params.in_file)
        params.out_dico = out_dico
        mapped_emb = self.mapping(Variable(out_emb, volatile=True)).data.cpu().numpy()

        all_keys = out_dico.word2id.keys()
        fhel = codecs.open(params.out_file)
        for key in all_keys:
            hv = mapped_emb[out_dico.index(key)]
            hv = hv / np.linalg.norm(hv)
            hv = map(str, list(hv))
            hv = " ".join([str(key)] + hv) + "\n"
            fhel.write(hv)

        fhel.close()
        logging.info("...Done!")
