import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy

from solution import RNN, GRU
from solution import make_model as TRANSFORMER


parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='RNN',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=10,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.8,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size,
                seq_len=args.seq_len, batch_size=10,
                vocab_size=10000, num_layers=args.num_layers,
                dp_keep_prob=args.dp_keep_prob)

#print(model.out_layer.weight.data)

#toy
#model.load_state_dict(torch.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=1_save_best_0/best_params.pt"))

#Trained RNN
model.load_state_dict(torch.load("RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=20_save_best_0"))

#Trained GRU
#model.load_state_dict(torch.load("GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=20_save_best_0"))

model.eval()

#print(model.out_layer.weight.data)

#also indices
inputs = torch.from_numpy(np.random.randint(1,high=10001,size=10).astype(np.int64))

hidden = model.init_hidden()
model.zero_grad()
hidden = repackage_hidden(hidden)
samples = model.generate(inputs, hidden, 20) #returns indices

#print(samples)

samples = samples.transpose(0,1)

filename = "data/ptb.train.txt"

def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

w2id, id2w = _build_vocab(filename)

output = []
for i in range(samples.size(0)):
    output.append([])
    for j in range(samples.size(1)):
        output[i].append(id2w[samples[i][j].item()])
    print(output[i])
