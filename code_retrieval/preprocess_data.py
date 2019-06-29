import os, sys, math, json
import numpy as np

sys.path.append('./preprocessing')
sys.path.append('./seq2seq')
from processor import Code_Intent_Pairs
from model import Seq2Seq
from data import get_train_loader, get_test_loader
# import torch
# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import LambdaLR


## TODO : should we rebuild the vocab?
process = ["combine_train_mined"]
if __name__ == "__main__":
    code_intent_pair = Code_Intent_Pairs()
    # code_intent_pair.load_dict(path='./vocab/')
    # mined_path = './corpus/mined.json'
    # # print(mined_path)
    # entries = code_intent_pair.load_raw_mined_data(mined_path)
    # proc_mined_path = '../processed_corpus/mined.json'
    # code_intent_pair.store_entries(proc_mined_path)

    ## we should probably use both mined and train for dictionary building.
    if "rebuild_dict" in process:
        code_intent_pair.get_dict_from_raw_combined(path_train="./corpus/train.json", \
                                                    path_mined="./corpus/mined.json")
        code_intent_pair.store_dict(path="./vocab/", mined=True)

    if "combine_train_mined" in process:
        # mined_path = './corpus/mined.json'
        code_intent_pair.load_dict(path='./vocab/', mined=True)
        entries = code_intent_pair.load_combined_data(path_train="./corpus/train.json", \
                                                      path_mined="./corpus/mined.json")
        code_intent_pair.store_entries("./processed_corpus/all.json")

        print(len(entries))
