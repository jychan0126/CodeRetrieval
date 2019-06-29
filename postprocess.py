import re
import json
import code_retrieval.preprocessing.tokenizer as token

f = open('generate_10.66.out')
# lines = f.readlines()
with open('code_retrieval/processed_corpus/test.json') as ft:
	data = json.load(ft)


pred = open('predict.out', 'w+')
tar = open('target.out', 'w+')

for i, line in enumerate(f.readlines()):
	
	# line_num = 287+4
	# print(lines[line_num-2])
	regular = re.match(r'[H]-(\d+)(\s\S+\s)(.+)', line)
	# print(regular)
	if regular == None:
		# print("NPOOOO")
		continue

	predict_code = regular.group(3)

	for key, value in data[int(regular.group(1))]['slot_map'].items():
		if key in predict_code:
			predict_code = predict_code.replace(key, value)
	
	pred.write(predict_code.replace('\n', '\\n') + '\n')
	tar.write(' '.join(token.tokenize_code(data[int(regular.group(1))]['code'])).replace('\n', '\\n') + '\n')
	# print(regular.group(3))

tar.close()
pred.close()
f.close()
'''
for i, line in enumerate(lines):
	print(i)
'''
'''
with open('code_retrieval/processed_corpus/test.json')as f:
	data = json.load(f)
	for key, value in data[int(test.group(1))]['slot_map'].items():
		if key in lines[line_num]:
			lines[line_num] = lines[line_num].replace(key, value)
	print(lines[line_num])
	print(' '.join(token.tokenize_code(data[int(test.group(1))]['code'])))
'''
