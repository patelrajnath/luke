CUDA_VISIBLE_DEVICES=1 python -m examples.cli --model-file=/RAJ/luke_base_500k.tar.gz --output-dir=model_original_run3 ner run --train-batch-size 2
#python -m examples.cli --model-file=/RAJ/luke_base_500k_kg_pretrain.tar.gz --output-dir=model ner run --train-batch-size 8

#python -m examples.cli --model-file=/RAJ/luke_base_500k.tar.gz --output-dir=model ner run --train-batch-size 8 --data-dir data/nlu
#python -m examples.cli --model-file=/RAJ/luke_base_500k_kg_pretrain.tar.gz --output-dir=model ner run --train-batch-size 8 --data-dir data/nlu
