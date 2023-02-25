import os
import logging
import nf

logger = logging.getLogger('args')
logger.addFilter(nf.fmt_filter)


def dir_path(dir_name) -> str:
    if os.path.isdir(dir_name):
        return dir_name
    else:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        raise NotADirectoryError(dir_name)


def common_dirs(parser) -> None:
    parser.add_argument(
        '-d', '--data_dir', help='Data output directory',
        type=dir_path, default='data'
    )
    parser.add_argument(
        '-m', '--models_dir', help='Models directory',
        type=dir_path, default='models'
    )


def device(parser) -> None:
    parser.add_argument(
        '-c', '--limit_cuda_device', help='Limit ops to specific cuda device.', type=int, default=None
    )


def train(parser) -> None:
    common_dirs(parser)
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
