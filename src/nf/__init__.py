import logging
import os
import sys

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


ner_tags = {
    'O': 0,
    'B-PER': 0, 'I-PER': 0,
    'B-LOC': 0, 'I-LOC': 0,
    'B-ORG': 0, 'I-ORG': 0,
    'B-MISC': 0, 'I-MISC': 0,
    'B-EVT': 0, 'I-EVT': 0,
    'B-PRO': 0, 'I-PRO': 0,
}


class Labeler:

    def __init__(self, remove_labels):
        label_id_map = {k: v for v, k in enumerate(ner_tags) if k not in self.remove_ner_tags}
        self.label_id_map = label_id_map
        ids_to_labels = {v: k for v, k in enumerate(ner_tags) if k not in self.remove_ner_tags}
        self.ids_to_labels = ids_to_labels

    def ids(self):
        pass