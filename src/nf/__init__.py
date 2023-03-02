import logging
import os
import sys
from typing import List, Dict

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

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: Dict[str, str] = None):
        if replace_labels is None:
            replace_labels = {}
        if labels is None:
            labels = []

        if file_name is not None and os.path.exists(file_name):
            with open(file_name, "r", encoding='utf-8') as fp:
                self._labels = fp.read().splitlines()
        else:
            self._labels = labels
        if not self._labels:
            raise ValueError('Either valid file_name or labels list must be present')
        self._source_labels = self._labels
        self._replace_labels = replace_labels
        self._label_to_id = {k: v for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}
        self._id_to_label = {v: k for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}

    def label2id(self, label: str) -> int:
        if not label:
            return -1
        if label in self._label_to_id:
            return self._label_to_id[label]
        return -1

    def labels2ids(self):
        return self._label_to_id

    def id2label(self, _id: int) -> str:
        if not _id:
            return None
        if _id in self._id_to_label:
            return self._id_to_label[_id]
        return None

    def ids2labels(self):
        return self._id_to_label

    def kept_labels(self):
        return self._label_to_id.keys()

    def labels(self):
        return self._label_to_id.keys()

    def source_labels(self):
        return self._source_labels

    def mun_labels(self):
        return len(self._label_to_id.keys())

    def filter_replace(self, text: str):
        for k, v in self._replace_labels.items():
            text = text.replace(k, v)
        return text


class MultiLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: List = None):
        super().__init__(file_name, labels, replace_labels)
        n = 2 ** len(self._source_labels)
        self._labels = []
        for i in range(0, n):
            bitlist = [k for k in range(i.bit_length()) if i & (1 << k)]
            label = ''
            for idx in bitlist:
                label += self._source_labels[idx]
            self._labels.append(label)
        self._label_to_id = {k: v for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}
        self._id_to_label = {v: k for v, k in enumerate(self._labels) if k not in self._replace_labels.keys()}

    def decode(self, idx: int):
        labels = []
        for i in range(0, idx.bit_length()):
            mask = 1 << i
            test = idx & mask
            if test != 0:
                labels.append(self._source_labels[i])
        return labels

    def binpowset(self, idx):
        labels = []
        for i in range(0, idx.bit_length()):
            mask = 1 << i
            test = idx & mask
            if test != 0:
                labels.append(1)
            else:
                labels.append(0)
        return labels
