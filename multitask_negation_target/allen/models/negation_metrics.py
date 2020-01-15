from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict
import re

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    bmes_tags_to_spans,
    TypedStringSpan,
)


@Metric.register("scope-tokens")
class  ScopeTokensMetric(Metric):

    """
    Implements a relaxed version of the scope tokens metric from StarSem 2012 
    shared task for negation detection. While the original metric required the 
    correct identification of the cue, here we only count the scopes.
    """

    def __init__(
                 self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 ) -> None:

        """

        """


        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ConfigurationError(
                    "Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'."
                )


        self._label_encoding = label_encoding
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label token counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prediction_map: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask : ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        prediction_map : ``torch.Tensor``, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            ``vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))``,
            rather than simply ``vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))``.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(
            predictions, gold_labels, mask, prediction_map
        )

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to SpanBasedF1Measure contains an "
                "id >= {}, the number of classes.".format(num_classes)
            )

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_string_labels = [
                self._label_vocabulary[label_id]
                for label_id in sequence_prediction[:length].tolist()
            ]
            gold_string_labels = [
                self._label_vocabulary[label_id]
                for label_id in sequence_gold_label[:length].tolist()
            ]


            # Scope tokens implementation
            for pred, gold in zip(predicted_string_labels,
                                  gold_string_labels):
                # only take into account the scopes, not the cues
                # Note this only works for the auxiliary tasks
                if (re.search(r"scope$", gold) or re.search(r"neg$", gold) or 
                    re.search(r"spec$", gold)):
                    if pred == gold:
                        self._true_positives[gold[2:]] += 1
                    elif pred != gold and pred == "O":
                        self._false_negatives[gold[2:]] += 1
                    elif pred != gold and gold == "O":
                        self._false_positives[pred[2:]] += 1
                elif re.search(r"cue$", gold):
                    if pred == gold:
                        self._true_positives[gold[2:]] += 1
                    elif pred != gold and pred == "O":
                        self._false_negatives[gold[2:]] += 1
                    elif pred != gold and gold == "O":
                        self._false_positives[pred[2:]] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float
        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "token-precision" + "-" + tag
            recall_key = "token-recall" + "-" + tag
            f1_key = "token-f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["token-precision-overall"] = precision
        all_metrics["token-recall-overall"] = recall
        all_metrics["token-f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
