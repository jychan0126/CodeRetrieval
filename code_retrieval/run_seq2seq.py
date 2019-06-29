import os, sys, math
import numpy as np
sys.path.append('./preprocessing')
sys.path.append('./seq2seq')
from processor import Code_Intent_Pairs
from model import Seq2Seq
from data import get_train_loader, get_test_loader
import torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import argparse
parser = argparse.ArgumentParser(description='Description of your program')
# parser.add_argument('-est','--est', dest='n_est', help='Description for foo argument', default=1000, type=int)
parser.add_argument('-nepoch','--nepoch', dest='nepoch', help='number of epoch to run', default=10, type=int)
parser.add_argument('-batchsize','--batchsize', dest='batchsize', help='batch size', default=16, type=int)
parser.add_argument('-data_mode', dest='data_mode', help="use only train set (train) or plus mined set (all) as training data", default="train", type=str)
args = parser.parse_args()


hyperP = {
    ## training parameters
    'batch_size': args.batchsize,
    'lr': 1e-3,
    'teacher_force_rate': 1.0,
    'max_epochs': args.nepoch,
    'lr_keep_rate': 0.97,  # set to 1.0 to not decrease lr overtime
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
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(1)

# if is_cuda:
#     torch.cuda.set_device(1)
print("is cuda : ", is_cuda)

def train(model, trainloader, optimizer, loss_f, hyperP):
    model.train()
    total_loss = 0
    loss_sum = 0
    total_correct = 0
    size = 0
    print_every = hyperP['print_every']
    accs = []
    for i, (inp_seq, original_out_seq, padded_out_seq, out_lens) in enumerate(trainloader):
        if is_cuda:
            inp_seq = [seq.to(device) for seq in inp_seq]
            padded_out_seq = padded_out_seq.to(device)
            original_out_seq = original_out_seq.to(device)
        # print(i)
        logits = model(inp_seq, padded_out_seq, out_lens)
        loss = loss_f(logits, original_out_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show stats
        # loss_sum += loss.item()
        # total_loss += loss.item()
        # _, predictions = torch.max(logits, dim=1)
        # total_correct += (predictions == original_out_seq).sum()
        # size += len(original_out_seq)
        # if (i + 1) % print_every == 0:
        #     print('Train: loss:{}\tacc:{}'.format(loss_sum / print_every, float(total_correct) / size), end='\r')
        #     accs.append(float(total_correct) / size)
        #     loss_sum = 0
        #     total_correct = 0
        #     size = 0
        # if (i + 1) % print_every == 0:
        #     print('Train: loss:{}'.format(loss_sum / print_every), end='\r')
            # accs.append(float(total_correct) / size)
            # loss_sum = 0
            # total_correct = 0
            # size = 0

    # return total_loss / len(trainloader)
    # return total_loss / len(trainloader), sum(accs)/len(accs)

if __name__ == "__main__":
    code_intent_pair = Code_Intent_Pairs()
    path = "vocab/"
    if args.data_mode == "train":
        code_intent_pair.load_dict(path, mined=False)
        train_path = "processed_corpus/train.json"
    elif args.data_mode == "all":
        code_intent_pair.load_dict(path, mined=True)
        train_path = "processed_corpus/all.json"

    special_symbols = code_intent_pair.get_special_symbols()
    word_size = code_intent_pair.get_word_size()
    code_size = code_intent_pair.get_code_size()
    # print('word size : ', word_size)
    # print('code size : ', code_size)
    # train_path = 'processed_corpus/train.json'
    # train_path = 'processed_corpus/all.json'
    if args.data_mode == "train":
        train_entries = code_intent_pair.load_entries(train_path)
    elif args.data_mode == "all":
        train_entries = code_intent_pair.load_entries(train_path, upper_bound=20000)
    code_intent_pair.pad()
    trainloader = get_train_loader(train_entries, special_symbols, hyperP)


    # define model
    model = Seq2Seq(word_size, code_size, hyperP)
    if is_cuda:
        model.to(device)
    if hyperP['load_pretrain_code_embed']:
        model.decoder.embed[0].load_state_dict(torch.load('./pretrain_code_lm/embedding-1556211835.t7'))
        if hyperP['freeze_embed']:
            for param in model.decoder.embed[0].parameters():
                param.requires_grad = False
    #%% md
    ### Training
    optimizer = optim.Adam(model.parameters(), lr=hyperP['lr'])
    loss_f = torch.nn.CrossEntropyLoss()
    lr_keep_rate = hyperP['lr_keep_rate']
    if lr_keep_rate != 1.0:
        lr_reduce_f = lambda epoch: lr_keep_rate ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_reduce_f)

    best_acc = 0.0
    losses = []
    teacher_force_rate = hyperP['teacher_force_rate']
    for e in range(hyperP['max_epochs']):
        print('run epoch ', e)
        # loss, accu = train(model, trainloader, optimizer, loss_f, hyperP)
        train(model, trainloader, optimizer, loss_f, hyperP)
        # losses.append(loss)
        # print('accuracy : ', accu)

        if lr_keep_rate != 1.0:
            scheduler.step()
        # change teacher force rate
        teacher_force_rate = max(0.7, 0.99 * teacher_force_rate)
        model.change_teacher_force_rate(teacher_force_rate)
        if e == 19:
            model.save('model_20.t7')
            print('model saved')
        elif e == 29:
            model.save('model_30.t7')
            print('model saved')
        elif e == 39:
            model.save('model_40.t7')
            print('model saved')
        elif e == 44:
            model.save('model_45.t7')
            print('model saved')
        elif e == 49:
            model.save('model_50.t7')
            print('model saved')

    teacher_force_rate = 0.7
    model.change_teacher_force_rate(teacher_force_rate)
    for e in range(50):
        loss = train(model, trainloader, optimizer, loss_f, hyperP)
        losses.append(loss)

        if e == 9:
            model.save('model_60.t7')
            print('model saved')
        elif e == 19:
            model.save('model_70.t7')
            print('model saved')
        elif e == 29:
            model.save('model_80.t7')
            print('model saved')
        elif e == 39:
            model.save('model_90.t7')
            print('model saved')
        elif e == 44:
            model.save('model_95.t7')
            print('model saved')
        elif e == 49:
            model.save('model_100.t7')
            print('model saved')