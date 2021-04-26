from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities

from examples.ner.eval import f1_score_span

seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
seq = ['I-PER', 'I-PER', 'O', 'I-LOC']
y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O'], ['B-PER', 'I-PER', 'O']]
print(get_entities(y_true))
print(get_entities(y_pred))

print(f1_score_span(y_true, y_pred))
print(f1_score(y_true, y_pred))

