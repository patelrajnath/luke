import json
import logging
import os
from argparse import Namespace

import click
import torch
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN
from .utils_model import load_examples, evaluate

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForNamedEntityRecognition

logger = logging.getLogger(__name__)


@click.group(name="ner")
def cli():
    pass


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/conll_2003", type=click.Path(exists=True))
@click.option("--do-train/--no-train", default=True)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=16)
@click.option("--max-seq-length", default=512)
@click.option("--no-entity-feature", is_flag=True)
@click.option("--no-word-feature", is_flag=True)
@click.option("--train-batch-size", default=32)
@click.option("--num-train-epochs", default=5.0)
@click.option("--seed", default=35)
@click.option("--train-on-dev-set", is_flag=True)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    common_args.update(task_args)
    args = Namespace(**common_args)
    set_seed(args.seed)
    # args.device = 'cpu'
    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.model_config.entity_vocab_size = 2
    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, _, processor = load_examples(args, "train")
    results = {}

    if args.do_train:
        model = LukeForNamedEntityRecognition(args, len(processor.get_labels(args.data_dir)))
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    if args.do_eval:
        model = LukeForNamedEntityRecognition(args, len(processor.get_labels(args.data_dir)))
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        dev_output_file = os.path.join(args.output_dir, "dev_predictions.txt")
        test_output_file = os.path.join(args.output_dir, "test_predictions.txt")
        results.update({f"dev_{k}": v for k, v in evaluate(args, model, "dev", dev_output_file).items()})
        results.update({f"test_{k}": v for k, v in evaluate(args, model, "test", test_output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results
