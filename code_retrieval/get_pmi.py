import os, sys, math
import numpy as np

sys.path.append('./preprocessing')
sys.path.append('./seq2seq')
from processor import Code_Intent_Pairs
from model import Seq2Seq, SimpleSeq2Seq
from data import get_train_loader, get_test_loader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from decoder import Decoder
from decoder import post_process_dummy, post_process_pmi
from evaluate import get_bleu_all, get_bleu_sent
from data import write_answer_json
import math

import argparse
parser = argparse.ArgumentParser(description='Description of your program')
# parser.add_argument('-est','--est', dest='n_est', help='Description for foo argument', default=1000, type=int)
parser.add_argument('-nepoch','--nepoch', dest='nepoch', help='number of epoch to run', default=10, type=int)
parser.add_argument('-batchsize','--batchsize', dest='batchsize', help='batch size', default=32, type=int)
parser.add_argument('-data_mode', dest='data_mode', help="use only train set (train) or plus mined set (all) as training data", default="train", type=str)
parser.add_argument('-atten', dest='atten', help='use attention or not', action='store_true')
parser.add_argument('-upper_bound','--upper_bound', dest='upper_bound', help='upper bound of all data mode', default=50000, type=int)
args = parser.parse_args()

class TrainSet():
    def __init__(self, code_intent_pair):
        self.code_intent_pair = code_intent_pair

    def __len__(self):
        return len(self.code_intent_pair)

    def __getitem__(self, idx):
        intent_idx = self.code_intent_pair[idx]['intent_indx']
        code_idx = self.code_intent_pair[idx]['code_indx_nocopy']
        return (intent_idx, code_idx)

if __name__ == "__main__":
    code_intent_pair = Code_Intent_Pairs()
    path = 'vocab/'
    code_intent_pair.load_dict(path)
    special_symbols = code_intent_pair.get_special_symbols()
    word_size = code_intent_pair.get_word_size()
    code_size = code_intent_pair.get_code_size()

    output_model_prefix = "./models/models_"
    if args.data_mode == "train":
        code_intent_pair.load_dict(path, mined=False)
        train_path = "processed_corpus/train.json"
        output_model_prefix = output_model_prefix + "train_"
    elif args.data_mode == "all":
        code_intent_pair.load_dict(path, mined=True)
        train_path = "processed_corpus/all.json"
        output_model_prefix = output_model_prefix + "all_"

    train_entries = code_intent_pair.load_entries(train_path)
    trainset = TrainSet(train_entries)

    pair_p = [[0] * code_size for i in range(word_size)]
    word_p = [0] * word_size
    code_p = [0] * code_size
    unit_p = 1.0 / len(trainset)

    for i, (intent, code) in enumerate(trainset):
        intent = set(intent)
        code = set(code)
        for word in intent:
            word_p[word] += unit_p
        for token in code:
            code_p[token] += unit_p

        for word in intent:
            for token in code:
                pair_p[word][token] += unit_p

    pmi = [[0] * code_size for i in range(word_size)]
    for i in range(word_size - 3):
        for j in range(code_size - 3):
            pmi[i][j] = math.log((pair_p[i][j] + 0.01 * unit_p) / (word_p[i] * code_p[j]))
            if pmi[i][j] < 1.5:
                pmi[i][j] = 0.0

    import pickle

    with open('./processed_corpus/pmi_matrix.bin', 'wb') as f:
        pickle.dump(pmi, f)