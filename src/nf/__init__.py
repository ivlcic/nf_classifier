import logging
import os
import sys
from typing import List, Dict
from datetime import datetime, timedelta

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


class BinaryLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: Dict[str, str] = None):
        super().__init__(file_name, labels, replace_labels)
        self._source_labels = self._labels
        self._replace_labels = replace_labels
        if replace_labels is not None:
            for k in self._replace_labels.keys():
                self._labels = self._labels.remove(k)
        if len(self._labels) != 1:
            raise ValueError('BinaryLabeler should have single label!')
        self._label_to_id = {self._labels[0]: 1}
        self._id_to_label = {1: self._labels[0]}


class MultiLabeler(Labeler):

    def __init__(self, file_name: str = None, labels: List = None, replace_labels: Dict[str, str] = None):
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


class DateIter(object):

    def __init__(self, start_date: datetime, end_date: datetime, step_sec: int = 1800):
        self._delta: timedelta = end_date - start_date
        self._start: datetime = start_date
        self._end: datetime = end_date
        if self._delta.days > 0:
            self._range_secs = range(0, (self._delta.days + 1) * (24 * 3600) + self._delta.seconds,  step_sec)
        else:
            if step_sec >= self._delta.seconds:
                step_sec = int(self._delta.seconds / 2)
            self._range_secs = range(0, self._delta.seconds, step_sec)
        self._range_pos: int = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._range_pos > 0:
            secs = self._range_secs[self._range_pos - 1]
            start = self._start + timedelta(seconds=secs)
            if start == self._end:
                raise StopIteration
        else:
            raise StopIteration

        try:
            secs = self._range_secs[self._range_pos]
            end = self._start + timedelta(seconds=secs)
            self._range_pos += 1
        except IndexError:
            end = self._end
            self._range_pos = 0

        return start, end
