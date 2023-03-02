import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import evaluate

from typing import List, Any
from torch.utils.data import DataLoader
from tokenizers.tokenizers import Encoding
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    BatchEncoding, PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer
from sklearn.metrics import classification_report, accuracy_score

import nf
import nf.data

logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)


class ModelContainer(torch.nn.Module):
    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(ModelContainer, self).__init__()

        self._labeler: nf.Labeler = labeler
        self._model: PreTrainedModel = None
        self._metric = None
        self._tokenizer: PreTrainedTokenizer = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_dir
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self._device = device

    def model(self):
        return self._model

    def labeler(self):
        return self._labeler

    def tokenizer(self):
        return self._tokenizer

    def metric(self):
        return self._metric

    def forward(self, input_id, mask, label):
        output = self._model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output


class TokenClassModelContainer(ModelContainer):

    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(TokenClassModelContainer, self).__init__(model_dir, model_name, labeler)

        self._metric = evaluate.load("seqeval")
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )
        self._model.to(self._device)

    def compute_metrics(self, p, test: bool = False):
        logits, labels_list = p

        # select predicted index with maximum logit for each token
        predictions_list = np.argmax(logits, axis=2)

        tagged_predictions_list = []
        tagged_labels_list = []
        for predictions, labels in zip(predictions_list, labels_list):
            tagged_predictions = []
            tagged_labels = []
            for pid, lid in zip(predictions, labels):
                if lid != -100:
                    tagged_predictions.append(self._labeler.id2label(pid))
                    tagged_labels.append(self._labeler.id2label(lid))
            tagged_predictions_list.append(tagged_predictions)
            tagged_labels_list.append(tagged_labels)

        results = self._metric.compute(
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


class SklearnClassificationReport:

    def compute(self, predictions: List[Any], references: List[Any], labels: List[str]):
        result = classification_report(references, predictions, zero_division=0, output_dict=True, target_names=labels)
        if 'accuracy' not in result:
            result['accuracy'] = accuracy_score(references, predictions)
        return result


class SeqClassModelContainer(ModelContainer):

    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(SeqClassModelContainer, self).__init__(model_dir, model_name, labeler)

        self._metric = SklearnClassificationReport()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )

    def compute_metrics(self, p, test: bool = False):
        logits, labels = p

        # select predicted index with maximum logit for each token
        pred = np.argmax(logits, axis=1)

        if isinstance(self._labeler, nf.MultiLabeler):
            decoded_predictions = []
            decoded_labels = []
            for p, l in zip(pred, labels):
                decoded_predictions.append(self._labeler.binpowset(p))
                decoded_labels.append(self._labeler.binpowset(l))
            pred = decoded_predictions
            labels = decoded_labels

        results = self._metric.compute(predictions=pred, references=labels, labels=self._labeler.source_labels())
        if test:
            return results
        logger.info("Batch eval: %s", results)
        if len(logger.handlers) > 0:
            logger.handlers[0].flush()
        return {
            "precision": results['macro avg']["precision"],
            "recall": results['macro avg']["recall"],
            "f1": results['macro avg']["f1-score"],
            "accuracy": results['accuracy']
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
                label_ids.append(self._labeler.label2id(labels[word_idx]))
        return label_ids

    def __init__(self, model: ModelContainer, data: pd.DataFrame, max_seq_len: int,
                 label_field: str = 'label', text_field: str = 'text'):
        """Encodes the text data and labels
        """
        self._model = model
        self._labeler = model.labeler()
        self.max_seq_len = max_seq_len

        ds_labels = [self._labeler.filter_replace(line).split() for line in data[label_field].values.tolist()]

        # check if labels in the dataset are also in labeler
        true_labels = self._labeler.kept_labels()
        unique_labels = set()
        for lb in ds_labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        if unique_labels != true_labels:
            logger.error("Unexpected label [%s] in [%s] in dataset!",
                         unique_labels, true_labels)
            exit(1)

        # encode the text
        texts = data[text_field].values.tolist()
        self.encodings: BatchEncoding = model.tokenizer()(
            texts, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="pt"
        )
        # encode the labels
        self.labels = []
        for i, e in enumerate(self.encodings.encodings):
            self.labels.append(self.align_labels(e, ds_labels[i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def train(args, mc: ModelContainer, result_path: str, data_path: str,
          label_field: str = 'label', text_field: str = 'text') -> None:
    training_args = TrainingArguments(
        output_dir=result_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        disable_tqdm=True,
        load_best_model_at_end=True,
        save_strategy='epoch',
        learning_rate=args.learn_rate,
        optim='adamw_torch',
        #optim='adamw_hf',
        save_total_limit=1,
        metric_for_best_model='f1',
        logging_strategy='epoch',
    )

    train_data, eval_data, test_data = nf.data.load_corpus(data_path)
    logger.debug("Constructing train data set [%s]...", len(train_data))
    train_set = DataSequence(mc, train_data, args.max_seq_len, label_field, text_field)
    logger.info("Constructed train data set [%s].", len(train_data))
    logger.debug("Constructing evaluation data set [%s]...", len(eval_data))
    eval_set = DataSequence(mc, eval_data, args.max_seq_len, label_field, text_field)
    logger.info("Constructed evaluation data set [%s].", len(eval_data))

    training_args.logging_steps = len(train_set)

    trainer = Trainer(
        model=mc.model(),
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=mc.tokenizer(),
        compute_metrics=mc.compute_metrics
    )
    logger.debug("Starting training...")
    trainer.train()
    logger.info("Training done.")
    logger.debug("Starting evaluation...")
    trainer.evaluate()
    logger.info("Evaluation done.")

    logger.info("Starting test set evaluation...")
    logger.debug("Constructing test data set [%s]...", len(test_data))
    test_set = DataSequence(mc, test_data, args.max_seq_len, label_field, text_field)
    logger.info("Constructed test data set [%s].", len(test_data))
    predictions, labels, _ = trainer.predict(test_set)
    results = mc.compute_metrics((predictions, labels), True)
    logger.info("Test set evaluation results:")
    logger.info("%s", results)
    combined_results = {}
    if os.path.exists(os.path.join(args.models_dir, 'results_all.json')):
        with open(os.path.join(args.models_dir, 'results_all.json')) as json_file:
            combined_results = json.load(json_file)
    combined_results[args.target_model_name] = results
    with open(os.path.join(args.models_dir, 'results_all.json'), 'wt', encoding='utf-8') as fp:
        json.dump(combined_results, fp, cls=nf.data.NpEncoder)
    with open(os.path.join(args.models_dir, args.target_model_name + ".json"), 'wt') as fp:
        json.dump(results, fp, cls=nf.data.NpEncoder)
