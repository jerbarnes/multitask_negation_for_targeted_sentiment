from typing import Iterable, List, Tuple, Optional, Callable, Any
from pathlib import Path
import re

def get_spans_labels(labels: List[str], 
                     filter_by_sentiment: Optional[str] = None
                     ) -> Tuple[List[Tuple[int, int]], List[str]]:
    '''
    :param labels: A List of labels for one text/sentence that contain labels in
                   the BIOUL format
    :param filter_by_sentiment: The sentiment label that all spans have to be
                                associated with, all other that do not have 
                                this sentiment label are not returned. If this 
                                is None then all spans regardless of sentiment 
                                label.
    :returns: A list of two tuples of equal length. 1. A list of Tuples where 
              each tuple contains two integer representing the index of the start 
              and end of a Target/Entity. 2. A list of sentiment values that 
              are associated with the Target/Entity indexes, the sentiment 
              value comes from the last tag in the index e.g. if the entity is 
              more than one token long and the sentiment associated to the first 
              token is different to the last token then the last token sentiment 
              is used.
    '''
    start_index = 0
    end_index = 0
    in_tag = False
    label_indexes = []
    label_values = []
    for index, label in enumerate(labels):
        if re.search(r'^B', label):
            if in_tag:
                raise ValueError(f'Should not have B labels during a tag {labels}')
            in_tag = True
            start_index = index
        elif re.search(r'^L', label):
            if not in_tag:
                raise ValueError(f'Should not have L labels not during a tag {labels}')

            end_index = index
            label_value = label.split('-')[1]
            if filter_by_sentiment is not None:
                if label_value == filter_by_sentiment:
                    label_values.append(label_value)
                    label_indexes.append((start_index, end_index))
            else:
                label_values.append(label_value)
                label_indexes.append((start_index, end_index))
            start_index, end_index = 0, 0
            in_tag = False
            
        elif re.search(r'^U', label):
            if in_tag:
                raise ValueError(f'Should not have U labels during a tag {labels}')

            label_value = label.split('-')[1]
            if filter_by_sentiment is not None:
                if label_value == filter_by_sentiment:
                    label_values.append(label_value)
                    label_indexes.append((index, index))
            else:
                label_values.append(label_value)
                label_indexes.append((index, index))
            
        elif re.search(r'^O', label):
            if index == 0:
                continue
            elif start_index != 0:
                raise ValueError(f'Should not have O labels during a tag {labels}')
        elif re.search(r'^I', label):
            if not in_tag:
                raise ValueError(f'Should only have I labels within a tag {labels}')
        else:
            raise ValueError(f'Unknown label {label} in {labels}')
    assert len(label_indexes) == len(set(label_indexes))
    assert len(label_values) == len(label_indexes)
    return label_indexes, label_values

def get_labels(result_fp: Path, gold: bool, run_number: Optional[int] = None
               ) -> Iterable[List[str]]:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param gold: Whether or not to return the gold labels.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :returns: Either the prediction labels or gold labels from the given results file.
    '''
    if gold and run_number is not None:
        raise ValueError(f'Cannot have both gold {gold} and run_number {run_number}')
    if not gold and run_number is None:
        raise ValueError('One has to be True or have a run number: Gold '
                         f'{gold} and run_number {type(run_number)}')
    with result_fp.open('r') as result_file:
        labels = []
        for line_index, line in enumerate(result_file):
            line = line.strip()
            if line:
                values = line.split()
                token = values[0]
                if gold:
                    gold = values[1]
                    labels.append(gold)
                elif run_number is not None:
                    try:
                        pred = values[2 + run_number]
                    except:
                        raise IndexError(f'{line} {line_index} {run_number}')
                    labels.append(pred)
            else:
                if labels:
                    yield labels
                labels = []
        if labels:
            yield labels

def get_gold_labels(result_fp: Path) -> Iterable[List[str]]:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :returns: All of the gold labels from the given results file.
    '''
    for labels in get_labels(result_fp, gold=True):
        yield labels

def get_prediction_labels(result_fp: Path, run_number: int) -> Iterable[List[str]]:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :returns: All of the prediction labels from the given results file.
    '''
    for labels in get_labels(result_fp, gold=False, run_number=run_number):
        yield labels

def get_gold_pred_labels(result_fp: Path, run_number: int, 
                         filter_by_sentiment: Optional[str] = None
                         ) -> Iterable[Tuple[List[Tuple[Tuple[int, int], str]],
                                             List[Tuple[Tuple[int, int], str]]]]:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :param filter_by_sentiment: If this contains a sentiment label then the 
                                the list of tuples will only contain targets 
                                that come from this sentiment label.
    :returns: A tuple of two where each of them contain the same items however 
              the 1. Are the gold labels, 2. Predicted labels. Each of the 
              tuples contain a list of tuples where each tuple contains the index
              of a target and the sentiment associated to that target. The index 
              relate to the tokens within the sentence/text. 
    '''
    for gold_labels, prediction_labels in zip(get_gold_labels(result_fp), 
                                              get_prediction_labels(result_fp, run_number)):
        gold_span_indexes, gold_sentiments = get_spans_labels(gold_labels, 
                                                              filter_by_sentiment=filter_by_sentiment)
        gold_combined = list(zip(gold_span_indexes, gold_sentiments))

        pred_span_indexes, pred_sentiments = get_spans_labels(prediction_labels, 
                                                              filter_by_sentiment=filter_by_sentiment)
        pred_combined = list(zip(pred_span_indexes, pred_sentiments))
        yield gold_combined, pred_combined

def get_correct_prediction_spans_labels(result_fp: Path, run_number: int) -> Tuple[List[str], List[str]]:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :returns: A two lists the 1. The list of gold sentiments, 2. The list of 
              predicted sentiments, each list will be of the same size. These list 
              are of sentiments where the predicted span was correct therefore 
              the number of sentiments returned should be equal to or less than 
              the total number of sentiments in the dataset. It will only be equal 
              if the model can predict the all the spans correctly e.g. perfect 
              recall or perfect F1. 
    '''
    gold_sentiments = []
    pred_sentiments = []
    for gold_combined, pred_combined in get_gold_pred_labels(result_fp, run_number, filter_by_sentiment=None):
        pred_span_sentiments = list(zip(*pred_combined)) if pred_combined else ([], [])
        pred_span_indexes = list(pred_span_sentiments[0]) 
        text_pred_sentiments = list(pred_span_sentiments[1])
        # Finding the TP and FN
        for gold_index, gold_sentiment in gold_combined:
            if gold_index in pred_span_indexes:
                sentiment_index = pred_span_indexes.index(gold_index)
                pred_sentiments.append(text_pred_sentiments[sentiment_index])
                gold_sentiments.append(gold_sentiment)
    num_gold_sentiments = len(gold_sentiments)
    num_pred_sentiments = len(pred_sentiments)
    len_assert_err = (f'The number of gold labels {num_gold_sentiments} should'
                      ' equal the number of correct predicted '
                      f'span sentiment labels {num_pred_sentiments}')
    assert num_pred_sentiments == num_gold_sentiments, len_assert_err
    return gold_sentiments, pred_sentiments

def span_f1(result_fp: Path, run_number: int, 
            filter_by_sentiment: Optional[str] = None, 
            ignore_sentiment: bool = False
            ) -> Tuple[float, float, float]:
    '''
    Span F1 exact match where an exact match is only True if the Span is correct 
    and the label/sentiment for that span is also correct. The label/sentiment for the 
    prediction is taken from the last token in the predicted span and is assumed 
    to represent the sentiment/label for the entire span.

    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :param filter_by_sentiment: The metrics returned represents are for extracting 
                                the given sentiment label.
    :param ignore_sentiment: If True then the scores are exact match of the 
                             span/target extraction task that does not take into 
                             account the sentiment value.
    :returns: A tuple of (Recall, Precision, F1)
    '''
    tp = 0
    num_pred = 0
    num_true = 0
    for gold_combined, pred_combined in get_gold_pred_labels(result_fp, run_number, filter_by_sentiment):
        pred_span_sentiments = list(zip(*pred_combined)) if pred_combined else ([], [])
        pred_span_indexes = list(pred_span_sentiments[0]) 
        pred_sentiments = list(pred_span_sentiments[1])
        # Finding the TP and FN
        for gold_index, gold_sentiment in gold_combined:
            if gold_index in pred_span_indexes:
                if ignore_sentiment:
                    tp += 1
                    continue
                sentiment_index = pred_span_indexes.index(gold_index)
                if pred_sentiments[sentiment_index] == gold_sentiment:
                    tp += 1
        num_pred += len(pred_combined)
        num_true += len(gold_combined)
    if num_true == 0:
        recall = 0.0
    else:
        recall = tp / num_true
    
    if num_pred == 0.0:
        precision = 0.0
    else:
        precision = tp / num_pred

    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))
    return recall, precision, f1

def span_label_metric(result_fp: Path, run_number: int, 
                      metric_func: Callable[[List[Any], List[Any]], float],
                      **metric_func_kwargs) -> float:
    '''
    :param result_fp: File path to the CONLL formatted results file where the 
                      format should be 
                      TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                      where the # represents a whitespace and each blank new 
                      line represents a different sentence/text.
    :param run_number: If the model has been ran multiple times then this will 
                       return the `run_number` result. Assume that each run 
                       has a separate column in the CONLL format where the run 
                       columns start on the second column 
                       e.g. TOKEN#GOLD LABEL#PREDICTION LABEL 0#PREDICTION LABEL 1 
                       where the # represents a whitespace.
    :param metric_func: Function to apply to the sentiment/labels e.g. Accuracy
                        or Macro F1 e.g. :py:func:`sklearn.metrics.f1_score`
    :param metric_func_kwargs: Keyword arguments to give to the `metric_func`
    :returns: The metric of the `metric_func` for the sentiment/labels of the 
              targets for targets that the predicted model got correct (thus 
              not necessarily all targets).
    '''
    gold, predicted = get_correct_prediction_spans_labels(result_fp, run_number)
    return metric_func(gold, predicted, **metric_func_kwargs)