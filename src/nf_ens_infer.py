#!/usr/bin/env python
import argparse
import logging
import os

import nf.args
from nf.torch import SeqClassModelContainer, train


logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='News Frames Multilabel Neural Classifier inference')

    parser.add_argument('-p', '--pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['fr_cul', 'fr_eco', 'fr_lab', 'fr_sec', 'fr_wel'], nargs="+", required=True)
    nf.args.common_dirs(parser, 'nf')
    nf.args.device(parser)
    # noinspection PyTypeChecker
    args = parser.parse_args()
    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    for model_path, model_name in zip(nf.args.pretrained_model_path(args), args.pretrained_model):
        mc = SeqClassModelContainer(
            model_path,
            model_name,
            nf.BinaryLabeler(labels=[model_name])
        )

    data_path = []
    for corpus in args.corpora:
        data_path.append(os.path.join(args.data_dir, corpus))
