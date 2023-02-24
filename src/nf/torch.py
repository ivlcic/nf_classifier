import nf
import logging
import numpy as np
import torch


from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding, \
    PreTrainedModel, PreTrainedTokenizer

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


class SeqClassModelContainer(ModelContainer):

    def __init__(self, model_dir: str, model_name: str, labeler: nf.Labeler):
        super(SeqClassModelContainer, self).__init__(model_dir, model_name, labeler)

        self.metric = evaluate.load("seqeval")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_dir, num_labels=len(labeler.label_id_map),
            id2label=labeler.ids_to_labels, label2id=labeler.label_id_map
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
