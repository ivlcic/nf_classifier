#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import nf
import nf.data
import nf.args
import argparse
import logging
import os


from nf.torch import DataSequence, TrainedModelContainer
from transformers import TrainingArguments, Trainer

logger = logging.getLogger('test')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Test')
    parser.add_argument('corpora', help='Corpora to use', nargs='+',
                        choices=[
                            'sl_500k', 'sl_bsnlp', 'sl_ewsd', 'sl_scr', 'sl',
                            'hr_500k', 'hr_bsnlp', 'hr',
                            'sr_set', 'sr',
                            'bs_wann', 'bs',
                            'mk_wann', 'mk',
                            'sq_wann', 'sq',
                            'cs_bsnlp', 'cs_cnec', 'cs',
                            'bg_bsnlp', 'bg',
                            'uk_bsnlp', 'uk',
                            'ru_bsnlp', 'ru',
                            'pl_bsnlp', 'pl',
                            'sk_bsnlp', 'sk_wann', 'sk'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for testing')
    nf.args.common_dirs(parser, context='ner')

    # noinspection PyTypeChecker
    args = parser.parse_args()
    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    mc: TrainedModelContainer = TrainedModelContainer(
        nf.args.pretrained_model_path(args, train=True),
        nf.Labeler(
            os.path.join(args.data_dir, 'tags.csv'),
            replace_labels=nf.args.replace_ner_tags(args)
        )
    )

    training_args = TrainingArguments(
        args.models_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
    )

    trainer = Trainer(
        model=mc.model(),
        args=training_args,
        tokenizer=mc.tokenizer(),
        compute_metrics=mc.compute_metrics
    )
    logger.info("Starting test set evaluation...")

    path_prefix = []
    for corpus in args.corpora:
        path_prefix.append(os.path.join(args.data_dir, corpus))

    _, _, test_data = nf.data.load_corpus(path_prefix)
    test_set = DataSequence(mc, test_data, args.max_seq_len)
    predictions, labels, _ = trainer.predict(test_set)
    results = mc.compute_metrics((predictions, labels), True)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
