#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import nf
import nf.args
import argparse
import collections
import logging
import os


from nf.torch import TrainedModelContainer

logger = logging.getLogger('export')
logger.addFilter(nf.fmt_filter)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ArgNamespace = collections.namedtuple(
    'ArgNamespace', [
        'pretrained_model', 'data_dir', 'models_dir', 'target_model', "limit_cuda_device"
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helper script to export model checkpoint to a directory.')
    parser.add_argument('pretrained_model', help='Pretrained model to export.')
    parser.add_argument('target_model', help='Target model (directory) name to export to.')
    nf.args.common_dirs(parser)
    nf.args.ner(parser)
    nf.args.device(parser)
    # noinspection PyTypeChecker
    args: ArgNamespace = parser.parse_args()

    if args.limit_cuda_device is not None:
        logger.info("Will run on specified cuda [%s] device only!", args.limit_cuda_device)

    mc = TrainedModelContainer(
        nf.args.pretrained_model_path(args, train=True),
        nf.Labeler(
            os.path.join(args.data_dir, 'tags.csv'),
            replace_labels=nf.args.replace_ner_tags(args)
        )
    )

    if os.sep in args.target_model:
        export_path = os.path.join(args.target_model)
    else:
        export_path = os.path.join(args.models_dir, args.target_model)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    logger.info("Exporting model and tokenizer:")
    logger.info("%s", mc.model().save_pretrained(export_path))
    logger.info("%s", mc.tokenizer().save_pretrained(export_path))
