import logging
import os
import sys
from typing import List

# sad hack
for x, arg in enumerate(sys.argv):
    if arg == '-c' and x + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[x + 1]
        break


def fmt_filter(record):
    record.levelname = '[%s]' % record.levelname
    record.funcName = '[%s]' % record.funcName
    record.lineno = '[%s]' % record.lineno
    return True


logging.basicConfig(
    format='%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addFilter(fmt_filter)


default_tmp_dir = 'tmp'
default_data_dir = 'data'
default_bsnlp_data_dir = 'data_bsnlp'
default_models_dir = 'models'
default_corpora_dir = 'corpora'


model_name_map = {
    'mcbert': 'bert-base-multilingual-cased',
    'xlmrb': 'xlm-roberta-base',
    'xlmrl': 'xlm-roberta-large'
}


def get_pretrained_model_path(args, train: bool = False) -> str:
    if train:
        pt_model_dir = os.path.join(default_tmp_dir, args.pretrained_model)
        if not os.path.exists(pt_model_dir):
            os.makedirs(pt_model_dir)
        return pt_model_dir
    else:
        return os.path.join(args.models_dir, args.pretrained_model)


class Labeler:

    def __init__(self, file_name: str = None, labels: List = None, remove_labels: List = None):
        if remove_labels is None:
            remove_labels = []
        if labels is None:
            labels = []

        if file_name is not None and os.path.exists(file_name):
            with open(file_name, "r", encoding='utf-8') as fp:
                self.labels = fp.read().splitlines()
        else:
            self.labels = labels
        if not self.labels:
            raise ValueError('Either valid file_name or labels list must be present')
        self.source_labels = self.labels
        self.removed_labels = remove_labels
        self.label_to_id = {k: v for v, k in enumerate(self.labels) if k not in self.removed_labels}
        self.id_to_label = {v: k for v, k in enumerate(self.labels) if k not in self.removed_labels}

    def label2id(self, label: str) -> int:
        if not label:
            return -1
        if label in self.label_to_id:
            return self.label_to_id[label]
        return -1

    def id2label(self, ident: int) -> str:
        if not ident:
            return None
        if ident in self.id_to_label:
            return self.id_to_label[ident]
        return None


class MultiLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, remove_labels: List = None):
        super().__init__(file_name, labels, remove_labels)
        n = 2 ** len(self.source_labels)
        self.labels = []
        for i in range(0, n):
            bitlist = [k for k in range(i.bit_length()) if i & (1 << k)]
            label = ''
            for idx in bitlist:
                label += self.source_labels[idx]
            self.labels.append(label)
