from collections import defaultdict

import seqeval.metrics
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from examples.ner.eval import f1_score_span, precision_score_span, recall_score_span
from examples.ner.utils import convert_examples_to_features, CoNLLProcessor
from luke.model import logger


def load_examples(args, fold, eval_with_loss=False):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = CoNLLProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    if fold == "train" and args.train_on_dev_set:
        examples += processor.get_dev_examples(args.data_dir)

    label_list = processor.get_labels(args.data_dir)

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples, label_list, args.tokenizer, args.max_seq_length, args.max_entity_length, args.max_mention_length
    )

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_start_positions=create_padded_sequence("entity_start_positions", 0),
            entity_end_positions=create_padded_sequence("entity_end_positions", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
        )
        if args.no_entity_feature:
            ret["entity_ids"].fill_(0)
            ret["entity_attention_mask"].fill_(0)

        if fold == "train" or eval_with_loss:
            ret["labels"] = create_padded_sequence("labels", -1)
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)

        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor


def compute_loss(args, model, fold):
    dataloader, examples, features, processor = load_examples(args, fold, eval_with_loss=True)
    total_loss = 0.0
    steps = 0
    for batch in tqdm(dataloader, desc="Eval"):
        steps += 1
        model.eval()
        inputs_with_labels = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**inputs_with_labels)
            loss = output[0]
        total_loss += loss.item()
    return total_loss / steps


def evaluate(args, model, fold, output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    label_list = processor.get_labels(args.data_dir)
    all_predictions = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            for j, span in enumerate(feature.original_entity_spans):
                if span is not None:
                    all_predictions[feature.example_index][span] = logits[i, j].detach().cpu().max(dim=0)

    assert len(all_predictions) == len(examples)

    final_labels = []
    final_predictions = []

    for example_index, example in enumerate(examples):
        predictions = all_predictions[example_index]
        doc_results = []
        for span, (max_logit, max_index) in predictions.items():
            if max_index != 0:
                doc_results.append((max_logit.item(), span, label_list[max_index.item()]))

        predicted_sequence = ["O"] * len(example.words)
        for _, span, label in sorted(doc_results, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        final_predictions += predicted_sequence
        final_labels += example.labels

    # convert IOB2 -> IOB1
    prev_type = None
    for n, label in enumerate(final_predictions):
        if label[0] == "B" and label[2:] != prev_type:
            final_predictions[n] = "I" + label[1:]
        prev_type = label[2:]

    if output_file:
        all_words = [w for e in examples for w in e.words]
        with open(output_file, "w", encoding='utf8') as f:
            for item in zip(all_words, final_labels, final_predictions):
                f.write(" ".join(item) + "\n")

    assert len(final_predictions) == len(final_labels)
    print("The number of labels:", len(final_labels))
    print(seqeval.metrics.classification_report(final_labels, final_predictions, digits=4))

    return dict(
        f1=seqeval.metrics.f1_score(final_labels, final_predictions),
        precision=seqeval.metrics.precision_score(final_labels, final_predictions),
        recall=seqeval.metrics.recall_score(final_labels, final_predictions),
        f1_span=f1_score_span(final_labels, final_predictions),
        precision_span=precision_score_span(final_labels, final_predictions),
        recall_span=recall_score_span(final_labels, final_predictions),
    )
