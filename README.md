### Parameter
---
TEXT=ir_data
python fairseq/preprocess.py --source-lang intent --target-lang code \
--trainpref $TEXT/train_tok_500 \
--validpref $TEXT/valid_tok_500 \
--testpref $TEXT/test_tok_500 \
--destdir data-bin/ir_data \
--nwordssrc 437 --nwordstgt 524 \
--joined-dictionary
===== v2
TEXT=ir_dataAdd_v3
python fairseq/preprocess.py --source-lang intent --target-lang code \
--srcdict $TEXT/dict.intent.txt \
--tgtdict $TEXT/dict.code.txt \
--trainpref $TEXT/train_tok_500 \
--validpref $TEXT/valid_tok_500 \
--testpref $TEXT/test_tok_500 \
--destdir data-bin/ir_data_intent2code_v3 \
--nwordssrc 437 --nwordstgt 524




# Train
CUDA_VISIBLE_DEVICES=1,2 python fairseq/train.py data-bin/ir_data    \
--arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
--lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096   \
--save-dir checkpoints/intent-code-base/  \
--no-progress-bar --log-format json --log-interval 50 --save-interval-updates  1000 \
--keep-interval-updates 20 --fp16 --max-update 30000 --max-epoch 150
===== v2
CUDA_VISIBLE_DEVICES=1,2 python fairseq/train.py data-bin/ir_data_v2    \
--arch transformer_vaswani_wmt_en_de_big \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
--lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096   \
--save-dir checkpoints/intent-code-base_v2/  \
--no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 \
--keep-interval-updates 20 --fp16 --max-update 30000 --max-epoch 200
===== v3
# Code to Intent
CUDA_VISIBLE_DEVICES=1,2 python fairseq/train.py data-bin/ir_data_v3 --source-lang code --target-lang intent   \
--arch transformer_vaswani_wmt_en_de_big \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
--lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096   \
--save-dir checkpoints/code-intent-base_v3/  \
--no-progress-bar --log-format json --log-interval 1000 \
--keep-interval-updates 20 --fp16 --max-update 30000 --max-epoch 200 \
--dropout 0.3 --save-interval 5

# Intent to Code
CUDA_VISIBLE_DEVICES=1,2 python fairseq/train.py data-bin/ir_data_intent2code_v3 --source-lang intent --target-lang code   \
--arch transformer_vaswani_wmt_en_de_big \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
--lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096   \
--save-dir checkpoints/intent-code-base_v3/  \
--no-progress-bar --log-format json --log-interval 1000 \
--keep-interval-updates 20 --fp16 --max-update 30000 --max-epoch 200 \
--dropout 0.3 --save-interval 5





# Average Checkpoints
python fairseq/scripts/average_checkpoints.py \
--inputs checkpoints/intent-code-base_v3/ \
--num-epoch-checkpoints  5 --output averaged_model_BT.pt


# Generate
CUDA_VISIBLE_DEVICES=0 python fairseq/generate.py \
data-bin/ir_data_intent2code_v3 --path averaged_model_BT.pt \
--remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
--max-len-a 1 --max-len-b 50|tee generate_BT.out


grep ^T generate_10.66.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate_10.66.ref

grep ^H generate_10.66.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate_10.66.sys



# Evaluate
python fairseq/score.py --sys predict.out --ref target.out


# Translation
python fairseq/interactive.py data-bin/ir_data_v3/ \
--path averaged_model.pt \
--beam 4



import torch
en2de_ensemble = torch.hub.load(
	'pytorch/fairseq',
	'transformer',
	model_name_or_path='transformer.wmt18.en-de',
	checkpoint_file='checkpoint154.pt:checkpoint153.pt:checkpoint154.pt:checkpoint154.pt:checkpoint154.pt',
	data_name_or_path='.',
	tokenizer='moses',
	aggressive_dash_splits=True,
	pe='subword_nmt',
)
