#!/usr/bin/env python
import argparse
import logging
import os

import nf
import nf.args
from nf.torch import SeqClassModelContainer, train


logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='News Frames Neural Classifier train')
    parser.add_argument('corpora', help='Corpora to use', nargs='+',
                        choices=[
                            'test'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    nf.args.train(parser, 'nf')
    # noinspection PyTypeChecker
    args = parser.parse_args()
    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    model_name = args.pretrained_model + '-' + '.'.join(args.corpora)
    args.target_model_name = model_name

    result_dir = os.path.join(args.models_dir, args.target_model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    mc = SeqClassModelContainer(
        nf.get_pretrained_model_path(args, True),
        nf.model_name_map[args.pretrained_model],
        nf.MultiLabeler(os.path.join(args.data_dir, 'tags.csv'))
    )
    data_path = []
    for corpus in args.corpora:
        data_path.append(os.path.join(args.data_dir, corpus))
    train(args, mc, result_dir, data_path, 'label', 'text')
