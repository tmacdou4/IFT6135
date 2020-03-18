import torch
import torch.nn as nn
from solution import *
import numpy as np
import collections

# net = RNN(emb_size=1000, hidden_size=10,
#                 seq_len=10, batch_size=5,
#                 vocab_size=10, num_layers=3,
#                 dp_keep_prob=.5)
#
# #print(net)
#
# i = torch.from_numpy(np.array([1,2,3,4,5]))
# h = torch.rand(3,5,10)
# n = 5
#
# #print(net.generate(i,h,n))
#
# net2 = GRU(emb_size=1000, hidden_size=10,
#                 seq_len=10, batch_size=5,
#                 vocab_size=10, num_layers=3,
#                 dp_keep_prob=0.5)
#
# print(net2.generate(i,h,n))

# module = MultiHeadedAttention(5, 25)
#
# q = torch.rand(10,20,25)
# k = torch.rand(10,20,25)
# v = torch.rand(10,20,25)
#
# print(module.forward(q,k,v))

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

print(id2w[6])

print(w2id["company"])
