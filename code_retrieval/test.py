import json
import pickle
import preprocessing.tokenizer as tokenizer


def load_dict(path=None, mined=False):
	if path == None:
		path = 'vocab/'
	if mined == False:
		word_dict_path = path + 'word_dict.bin'
		code_dict_path = path + 'code_dict.bin'
	else:
		word_dict_path = path + 'word_dict_all.bin'
		code_dict_path = path + 'code_dict_all.bin'
	num2word = pickle.load(open(word_dict_path, 'rb'))
	word2num = dict(zip(num2word, range(0, len(num2word))))
	print(word2num)

	print("======")
	num2code = pickle.load(open(code_dict_path, 'rb'))
	code2num = dict(zip(num2code, range(0, len(num2code))))
	print(code2num)



def Write_data(content, dataset):
	with open('processed_corpus/'+dataset+'.json', 'r') as f:
		data = json.load(f)

	f = open('../ir_data/'+dataset+'_tok_500.'+content, 'w+')

	for l in range(len(data)):
		s = str(data[l]).split('\'')
		intent = []
		for i in range(len(s)):
			if s[i]==content:
				k=2
				while i+k < len(s):
					if s[i+k] != '], ':
						if s[i+k] != ', ':
							intent.append(s[i+k])
							#traipn print(s[i+k])
					else:
						break
					k += 1
				break
		if l == len(data)-1:
			f.write(" ".join(intent))
			break
		f.write(" ".join(intent)+"\n")
	f.close()


# Write_data("code", "valid")
def Data(dataType, lang):
	with open('processed_corpus/'+dataType+'.json', 'r') as f:
		data = json.load(f)
		#dic = json.loads(data[0])
	f = open('../ir_data/'+dataType+'_tok_500.'+lang, 'w+')
	# print(data[98]["code"].replace("in","\\n"))
	
	for i in range(len(data)):
		if i == len(data)+1:
			f.write(data[i][lang])
		else:
			f.write(str(data[i][lang]).replace("\n", "\\n")+"\n")
	
	f.close()

# Data("test", "code")

typ = 'train'
with open('processed_corpus/'+typ+'.json', 'r') as f:
	data = json.load(f)
f = open('../ir_data_v2/'+typ+'_tok_500.code', 'w+')
f_int = open('../ir_data_v2/'+typ+'_tok_500.intent', 'w+')

for i in range(len(data)):
	if i == len(data)-1:
		f.write(' '.join(data[i]["code"]).replace('\n', '\\n'))
		f_int.write(' '.join(data[i]["intent"]))
	else:
		f.write(' '.join(data[i]["code"]).replace('\n', '\\n')+'\n')
		f_int.write(' '.join(data[i]["intent"]) + '\n')

f.close()
f_int.close()

with open('processed_corpus/test.json', 'r') as f:
	test_data = json.load(f)
f = open('../ir_data_v2/test_tok_500.code', 'w+')
f_int = open('../ir_data_v2/test_tok_500.intent', 'w+')

for i in range(len(test_data)):
	code = tokenizer.tokenize_code(test_data[i]["code"])
	intent = tokenizer.tokenize_intent(test_data[i]["intent"])

	if i == len(test_data)-1:
		f.write(' '.join(code).replace('\n', '\\n'))
		f_int.write(' '.join(intent[0]))
	else:
		f.write(' '.join(code).replace('\n', '\\n') + '\n')
		f_int.write(' '.join(intent[0]) + '\n')

# print(tokenizer.tokenize_code(test_data[0]["code"]))
