import io

fi = open('ir_dataAdd_v3/train_tok_500.intent', 'a+')

with open('code2intent_generate.ref') as f:
	for line in f.readlines():
		fi.write(line)

fi.close()	
