python luke/cli.py build-wikipedia-pretraining-dataset \
	wiki/wiki_vocab roberta-large \
	wiki/wiki_vocab_entity \
	wiki/ \
	--max-num-documents 1000000
