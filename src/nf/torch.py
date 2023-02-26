from typing import List

import nf
import logging
import numpy as np
import pandas as pd
import torch
import evaluate

from torch.utils.data import DataLoader
from tokenizers.tokenizers import Encoding
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    BatchEncoding, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)


class ModelContainer(torch.nn.Module):
    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(ModelContainer, self).__init__()

        self.labeler = labeler
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_dir
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(device)
        self.device = device


class TokenClassModelContainer(ModelContainer):

    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(TokenClassModelContainer, self).__init__(model_dir, model_name, labeler)

        self.metric = evaluate.load("seqeval")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

    def compute_metrics(self, p, test: bool = False):
        predictions_list, labels_list = p

        # select predicted index with maximum logit for each token
        predictions_list = np.argmax(predictions_list, axis=2)

        tagged_predictions_list = []
        tagged_labels_list = []
        for predictions, labels in zip(predictions_list, labels_list):
            tagged_predictions = []
            tagged_labels = []
            for pid, lid in zip(predictions, labels):
                if lid != -100:
                    tagged_predictions.append(self.ids_to_labels[pid])
                    tagged_labels.append(self.ids_to_labels[lid])
            tagged_predictions_list.append(tagged_predictions)
            tagged_labels_list.append(tagged_labels)

        results = self.metric.compute(
            predictions=tagged_predictions_list, references=tagged_labels_list, scheme='IOB2', mode='strict'
        )
        if test:
            return results
        logger.info("Batch eval: %s", results)
        if len(logger.handlers) > 0:
            logger.handlers[0].flush()
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


class SeqClassModelContainer(ModelContainer):

    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(SeqClassModelContainer, self).__init__(model_dir, model_name, labeler)

        self.metric = evaluate.load("seqeval")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )

    def forward(self, input_id, mask, label):
        output = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

    def compute_metrics(self, p, test: bool = False):
        predictions_list, labels_list = p

        # select predicted index with maximum logit for each token
        predictions_list = np.argmax(predictions_list, axis=2)

        tagged_predictions_list = []
        tagged_labels_list = []
        for predictions, labels in zip(predictions_list, labels_list):
            tagged_predictions = []
            tagged_labels = []
            for pid, lid in zip(predictions, labels):
                if lid != -100:
                    tagged_predictions.append(self.ids_to_labels[pid])
                    tagged_labels.append(self.ids_to_labels[lid])
            tagged_predictions_list.append(tagged_predictions)
            tagged_labels_list.append(tagged_labels)

        results = self.metric.compute(
            predictions=tagged_predictions_list, references=tagged_labels_list, scheme='IOB2', mode='strict'
        )
        if test:
            return results
        logger.info("Batch eval: %s", results)
        if len(logger.handlers) > 0:
            logger.handlers[0].flush()
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


class TrainedModelContainer(ModelContainer):
    def __init__(self, model_dir: str, labeler: nf.Labeler):
        super().__init__(None, model_dir, labeler)


class DataSequence(torch.utils.data.Dataset):

    def align_labels(self, encoded: Encoding, labels: List[str]):
        word_ids = encoded.word_ids
        label_ids = []
        max_idx = len(labels)
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx < 0 or word_idx >= max_idx:
                label_ids.append(-100)
            else:
                label_ids.append(self.labeler.label2id(labels[word_idx]))
        return label_ids

    def __init__(self, model: ModelContainer, data: pd.DataFrame, max_seq_len: int,
                 label_field: str = 'label', text_field: str = 'text'):
        """Encodes the text data and labels
        """
        self.model = model
        self.labeler = model.labeler
        self.max_seq_len = max_seq_len

        ds_labels = [self.labeler.filter_replace(line).split() for line in data[label_field].values.tolist()]

        # check if labels in the dataset are also in labeler
        true_labels = self.labeler.kept_labels()
        unique_labels = set()
        for lb in ds_labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        if unique_labels != true_labels:
            logger.error("Unexpected label [%s] in [%s] in dataset!",
                         unique_labels, true_labels)
            exit(1)

        texts = data[text_field].values.tolist()
        self.encodings: BatchEncoding = model.tokenizer(
            texts, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="pt"
        )
        self.labels = []
        for i, e in enumerate(self.encodings.encodings):
            self.labels.append(self.align_labels(e, all_labels[i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
