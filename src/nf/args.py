import os
import logging
from typing import Dict, Any

import nf

logger = logging.getLogger('args')
logger.addFilter(nf.fmt_filter)


def chech_param(conf: Dict, p_name: str) -> Any:
    p = conf.get(p_name)
    if not p:
        logger.warning('Missing [%s] param in [%s] config', p_name, conf)
        exit(1)
    return p


def chech_dir_param(conf: Dict, param_name: str, parent_path: str) -> str:
    fname = chech_param(conf, param_name)
    fpath = os.path.join(parent_path, fname)
    if not os.path.exists(fpath):
        logger.warning('Missing [%s] filename in dir [%s]', fname, parent_path)
        exit(1)
    return fpath


def dir_path(dir_name) -> str:
    if os.path.isdir(dir_name):
        return dir_name
    else:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        raise NotADirectoryError(dir_name)


def common_dirs(parser, context: str = None) -> None:
    data = nf.default_data_dir
    models = nf.default_models_dir
    if context:
        data = os.path.join(data, context)
        models = os.path.join(models, context)
    parser.add_argument(
        '-d', '--data_dir', help='Data output directory',
        type=dir_path, default=data
    )
    parser.add_argument(
        '-m', '--models_dir', help='Models directory',
        type=dir_path, default=models
    )


def device(parser) -> None:
    parser.add_argument(
        '-c', '--limit_cuda_device', help='Limit ops to specific cuda device.', type=int, default=None
    )


def train(parser, context: str = None) -> None:
    common_dirs(parser, context)
    parser.add_argument(
        '-b', '--batch', help='Batch size.', type=int, default=32
    )
    parser.add_argument(
        '-l', '--learn_rate', help='Learning rate', type=float, default=5e-5
    )
    parser.add_argument(
        '-e', '--epochs', help='Number of epochs.', type=int, default=20
    )
    device(parser)


def ner(parser) -> None:
    parser.add_argument(
        '--max_seq_len', help='Max sentence length in tokens / words.', type=int, default=256
    )
    parser.add_argument(
        '--no_misc', help='Remove MISC tag (replace i with "O").', action='store_true', default=False
    )
    parser.add_argument(
        '--pro', help='Enable Product (PRO) tag.', action='store_true', default=False
    )
    parser.add_argument(
        '--evt', help='Enable Event (EVT) tag.', action='store_true', default=False
    )


def replace_ner_tags(args) -> Dict[str, str]:
    del_misc = {}
    if hasattr(args, 'no_misc') and args.no_misc:
        del_misc['B-MISC'] = 'O'
        del_misc['I-MISC'] = 'O'
    if not hasattr(args, 'pro') or not args.pro:
        del_misc['B-PRO'] = 'O'
        del_misc['I-PRO'] = 'O'
    if not hasattr(args, 'evt') or not args.evt:
        del_misc['B-EVT'] = 'O'
        del_misc['I-EVT'] = 'O'
    return del_misc
