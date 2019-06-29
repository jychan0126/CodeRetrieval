import os, sys, math
import numpy as np

sys.path.append('./preprocessing')
sys.path.append('./seq2seq')
from processor import Code_Intent_Pairs
from model import Seq2Seq
from data import get_train_loader, get_test_loader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from decoder import Decoder
from decoder import post_process_dummy, post_process_pmi
from evaluate import get_bleu_all, get_bleu_sent
from data import write_answer_json


hyperP = {
    ## training parameters
    'batch_size': 32,
    'lr': 1e-3,
    'teacher_force_rate': 0.90,
    'max_epochs': 50,
    'lr_keep_rate': 0.95,  # set to 1.0 to not decrease lr overtime
    'load_pretrain_code_embed': False,
    'freeze_embed': False,

    ## encoder architecture
    'encoder_layers': 2,
    'encoder_embed_size': 128,
    'encoder_hidden_size': 384,
    'encoder_dropout_rate': 0.3,

    ## decoder architecture
    'decoder_layers': 2,
    'decoder_embed_size': 128,
    'decoder_hidden_size': 384,
    'decoder_dropout_rate': 0.3,

    ## attn architecture
    'attn_hidden_size': 384,

    ## visualization
    'print_every': 10,
}

is_cuda = torch.cuda.is_available()
print('is cuda : ', is_cuda)
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    code_intent_pair = Code_Intent_Pairs()
    path = 'vocab/'
    code_intent_pair.load_dict(path)
    special_symbols = code_intent_pair.get_special_symbols()
    word_size = code_intent_pair.get_word_size()
    code_size = code_intent_pair.get_code_size()

    test_path = 'processed_corpus/test.json'
    test_entries = code_intent_pair.load_entries(test_path)
    testloader = get_test_loader(test_entries)

    model = Seq2Seq(word_size, code_size, hyperP)
    if hyperP['load_pretrain_code_embed']:
        model.decoder.embed[0].load_state_dict(torch.load('./pretrain_code_lm/embedding-1556211835.t7'))
        if hyperP['freeze_embed']:
            for param in model.decoder.embed[0].parameters():
                param.requires_grad = False

    model.load('model_100.t7')
    beam_decoder = Decoder(model)
    if is_cuda:
        model.to(device)
        # beam_decoder.to(device)
    model.eval()

    # input('check gpu location')
    sos = special_symbols['code_sos']
    eos = special_symbols['code_eos']
    unk = special_symbols['code_unk']
    idx2code = code_intent_pair.idx2code
    intent2idx = code_intent_pair.intent2idx

    dummy_code_list = []
    true_code_list = []

    for i, (src_seq, slot_map, code, intent) in enumerate(testloader):
        if is_cuda:
            src_seq = [seq.to(device) for seq in src_seq]
            # slot_map = slot_map.to(device)
            # original_out_seq = original_out_seq.to(device)
        beams = beam_decoder.decode(src_seq, sos, eos, unk, beam_width=3)
        dummy_code = post_process_dummy(slot_map, beams, idx2code)
        dummy_code_list.append(dummy_code)
        true_code_list.append(code)
    for i, (src_seq, slot_map, code, intent) in enumerate(testloader):
        print('intent : ', intent)
        print('gt : ', code)
        print(dummy_code_list[i])
        if i == 10:
            break

    bleu = get_bleu_all(dummy_code_list, true_code_list)
    print('bleu scores : ', bleu)

    write_answer_json(dummy_code_list)