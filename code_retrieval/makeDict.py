import pickle
from fairseq import tasks, options
import json
'''
parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)

task = tasks.setup_task(args)
'''
with open('vocab/word_dict.bin', 'rb') as f:
	d = pickle.load(f)
'''
with open('../data-bin/ir_data/test.intent-code.intent.bin', 'r', encoding='utf-8') as f:
	f.readline()
'''
with open('processed_corpus/train.json', 'r') as f:
	train_data = json.load(f)
with open('processed_corpus/valid.json', 'r') as f:
	valid_data = json.load(f)
'''
with open('processed_corpus/test.json', 'r') as f:
	test_data = json.load(f)
print(test_data[len(test_data)-1]['intent'].split(' ')[0])
print(len(test_data[len(test_data)-1]['intent_indx']))
'''
dic = {}
typ = 'intent'

for i in range(len(train_data)):
	for j in range(len(train_data[i][typ])): 
		dic[train_data[i][typ][j]] = train_data[i]['intent_indx'][j]
''' # Don't put valid data in dict
for i in range(len(valid_data)):
	for j in range(len(valid_data[i]['code'])): 
		dic[valid_data[i]["code"][j]] = valid_data[i]['code_indx_copy'][j]
'''

'''
for i in range(len(test_data)):
	for j in range(len(test_data[i]['intent'].split(' '))): 
		dic[test_data[i]["intent"].split(' ')[0]] = test_data[i]['intent_indx'][j]
'''
f = open('../ir_data_v3/dict.'+typ+'.txt', 'w+')
for text, num in dic.items():
	f.write(text.replace('\n', '\\n') + ' ' + str(num) + '\n')

f.close()
print(len(dic))
