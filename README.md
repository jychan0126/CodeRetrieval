# Description
## generateConcate.py
Concate two txt. When using Back-translation, we need to concate the synthetic parallel corpus and original bitext.

## postprocess.py
Process test.json to match the format of the output file. 


# Parameter

## Pre-process
```
TEXT=data \
python fairseq/preprocess.py --source-lang intent --target-lang code \
--srcdict $TEXT/dict.intent.txt \
--tgtdict $TEXT/dict.code.txt \
--trainpref $TEXT/train_tok_500 \
--validpref $TEXT/valid_tok_500 \
--testpref $TEXT/test_tok_500 \
--destdir data-bin/data \
--nwordssrc 437 --nwordstgt 524 \
```

## Train
```
CUDA_VISIBLE_DEVICES=1,2 python fairseq/train.py data-bin/data    \
--arch transformer_vaswani_wmt_en_de_big \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
--lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096   \
--save-dir checkpoints/intent-code-base/  \
--no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 \
--keep-interval-updates 20 --fp16 --max-update 30000 --max-epoch 200
```

## Average Checkpoints
```
python fairseq/scripts/average_checkpoints.py \
--inputs checkpoints/intent-code-base/ \
--num-epoch-checkpoints  5 --output averaged_model.pt
```

## Generate
```
CUDA_VISIBLE_DEVICES=0 python fairseq/generate.py \
data-bin/ir_data_intent2code_v3 --path averaged_model.pt \
--remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
--max-len-a 1 --max-len-b 50|tee generate.out
```

## Grab the output
```
grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref \
grep ^H generate.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys \
```

## Evaluate
```
python fairseq/score.py --sys generate.sys --ref generate.ref
```

## Interactive Translation
```
python fairseq/interactive.py data-bin/data/ \
--path averaged_model.pt \
--beam 4
```

# Reference
[Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)
