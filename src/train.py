import argparse
import collections
import logging
import os

import nf
import nf.args
from nf.torch import SeqClassModelContainer

from transformers import TrainingArguments, Trainer

logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'lang', 'corpora', 'pretrained_model', 'data_dir', 'models_dir', 'learn_rate', 'epochs',
        'data_split', 'non_reproducible_shuffle', 'batch', 'max_seq_len', 'no_misc', "limit_cuda_device",
        "target_model_name"
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='News Frames Neural Classifier train')
    parser.add_argument('corpora', help='Corpora to use', nargs='+',
                        choices=[
                            'test'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    nf.args.train(parser)
    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()
    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    model_name = args.pretrained_model + '-' + '.'.join(args.corpora)
    args.target_model_name = model_name

    model_result_dir = os.path.join(args.models_dir, args.target_model_name)
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    training_args = TrainingArguments(
        output_dir=model_result_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        disable_tqdm=True,
        load_best_model_at_end=True,
        save_strategy='epoch',
        learning_rate=args.learn_rate,
        optim='adamw_torch',
        save_total_limit=1,
        metric_for_best_model='f1',
        logging_strategy='epoch',
    )

    mc = SeqClassModelContainer(
        nf.get_pretrained_model_path(args, True),
        nf.model_name_map[args.pretrained_model],
        []
    )
    path_prefix = []
    for corpus in args.corpora:
        path_prefix.append(os.path.join(args.data_dir, corpus))