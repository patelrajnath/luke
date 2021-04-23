#!/bin/bash 

for m in luke_base_500k.tar.gz luke_base_500k_kg_pretrain_conll_ner.tar.gz luke_base_500k_kg_pretrain_kaggle_ner.tar.gz;
do
        for n in 1 2 3;
        do
		filename="${m%%.*}"
		echo $m
		echo $filename
		echo $n
		CUDA_VISIBLE_DEVICES=0 python -m examples.cli \
			--model-file=/RAJ/${m} \
			--output-dir=model_${filename}_run-${n} ner run \
			--train-batch-size 2
        done
done

#CUDA_VISIBLE_DEVICES=1 python -m examples.cli --model-file=/RAJ/luke_base_500k.tar.gz --output-dir=model_original_run3 ner run --train-batch-size 2
#CUDA_VISIBLE_DEVICES=1 python -m examples.cli --model-file=/RAJ/luke_base_500k_kg_pretrain.tar.gz --output-dir=model_conll_pretrain-kg ner run --train-batch-size 2
#CUDA_VISIBLE_DEVICES=1 python -m examples.cli --model-file=/RAJ/luke_base_500k.tar.gz --output-dir=model_nlu ner run --train-batch-size 2 --data-dir data/nlu
#CUDA_VISIBLE_DEVICES=1 python -m examples.cli --model-file=/RAJ/luke_base_500k_kg_pretrain.tar.gz --output-dir=model ner run --train-batch-size 8 --data-dir data/nlu
