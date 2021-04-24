from seqeval.metrics.sequence_labeling import get_entities
seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
seq = ['I-PER', 'I-PER', 'O', 'I-LOC']
y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
print(get_entities(y_true))
print(get_entities(y_pred))
