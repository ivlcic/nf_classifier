#!/usr/bin/env python
import nf
import nf.torch
import nf.args
import nf.data
import os
import logging
import argparse

from nf.torch import TokenClassModelContainer, train

logger = logging.getLogger('ner_train')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Neural train for Slovene, Croatian and Serbian language')
    parser.add_argument('corpora', help='Corpora to use', nargs='+',
                        choices=[
                            'sl_500k', 'sl_bsnlp', 'sl_ewsd', 'sl_scr', 'sl',
                            'hr_500k', 'hr_bsnlp', 'hr',
                            'sr_set', 'sr',
                            'bs_wann', 'bs',
                            'mk_wann', 'mk',
                            'sq_wann', 'sq',
                            'cs_bsnlp', 'cs_cnec', 'cs'
                            'bg_bsnlp', 'bg',
                            'uk_bsnlp', 'uk',
                            'ru_bsnlp', 'ru',
                            'pl_bsnlp', 'pl'
                        ])
    parser.add_argument('pretrained_model', help='Pretrained model to use for fine tuning',
                        choices=['mcbert', 'xlmrb', 'xlmrl'])
    nf.args.train(parser, 'ner')
    nf.args.ner(parser)

    # noinspection PyTypeChecker
    args = parser.parse_args()

    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    model_name = args.pretrained_model + '-' + '.'.join(args.corpora)
    if args.no_misc:
        model_name += '-nomisc'
    args.target_model_name = model_name

    model_result_dir = os.path.join(args.models_dir, args.target_model_name)
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    mc = TokenClassModelContainer(
        nf.get_pretrained_model_path(args, True),
        nf.model_name_map[args.pretrained_model],
        nf.Labeler(os.path.join(
            args.data_dir, 'tags.csv'),
            replace_labels=nf.args.replace_ner_tags(args)
        )
    )
    path_prefix = []
    for corpus in args.corpora:
        path_prefix.append(os.path.join(args.data_dir, corpus))
    train(args, mc, model_result_dir, path_prefix)
