from typing import List, Dict, Any
from pathlib import Path
import re
from target_extraction.data_types import TargetTextCollection, TargetText

def from_bioul_to_bio(bioul: List[str], remove_tag_labels: bool = False
                      ) -> List[str]:
    '''
    :param biose_fp: File path to the data that is in CONLL like format: 
                     TOKEN LABEL\n where sentences are split by empty new lines.
                     The label format is in BIOSE = Beginning of, inside of, 
                     outside, single unit, and end of.
    :param bio_fp: File path to save the data that is in `biose_fp` to 
                   this file but in BIO format where S tags will become B tags 
                   and E tags will become I tags.
    '''
    bio_labels = []
    for label in bioul:
        if re.search(r'^L', label):
            label = re.sub(r'^L', 'I', label)
        elif re.search(r'^U', label):
            label = re.sub(r'^U', 'B', label)
        if remove_tag_labels:
            label = label.split('-')[0]
        bio_labels.append(label)
    assert len(bioul) == len(bio_labels)
    return bio_labels


def get_sentiment_labels_from_sequence(sequence_labels: List[str]) -> List[str]:
    '''
    Assumes the sequence labels are in BIOUL format
    '''
    sentiment_labels: List[str] = []
    for sequence_label in sequence_labels:
        if re.search(r'^B', sequence_label):
            continue
        elif re.search(r'^I', sequence_label):
            continue
        elif re.search(r'^L', sequence_label):
            sentiment_labels.append(sequence_label.split('-')[1])
        elif re.search(r'^U', sequence_label):
            sentiment_labels.append(sequence_label.split('-')[1])
        elif re.search(r'^O', sequence_label):
            continue
        else:
            raise ValueError('Sequence labels should be `B` `I` `U` `L` or `O` '
                             f'and not {sequence_label}.')
    return sentiment_labels

def create_target_text(sentences: List[str], num_results: int, _id: int, 
                       result_structs: List[List[TargetText]]) -> List[List[TargetText]]:
    tokens = []
    gold_labels = []
    pred_labels = []
    for sentence_index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        sentence_elements = sentence.split()
        assert len(sentence_elements) == num_results + 2
        tokens.append(sentence_elements[0])
        gold_labels.append(sentence_elements[1])
        temp_pred_labels = sentence_elements[2:]
        if sentence_index == 0:
            for pred_label in temp_pred_labels:
                pred_labels.append([pred_label])
        else:
            for pred_index, pred_label in enumerate(temp_pred_labels):
                pred_labels[pred_index].append(pred_label)
    text = ' '.join(tokens)
    gold_sentiments = get_sentiment_labels_from_sequence(gold_labels)
    gold_labels = from_bioul_to_bio(gold_labels, remove_tag_labels=True)
    gold_target_text = TargetText.target_text_from_prediction(text=text, text_id=_id, tokenized_text=tokens, sequence_labels=gold_labels)
    gold_spans = gold_target_text['spans']

    pred_sentiments = [get_sentiment_labels_from_sequence(pred_label) for pred_label in pred_labels]
    pred_labels = [from_bioul_to_bio(pred_label, remove_tag_labels=True) for pred_label in pred_labels]
    pred_spans = []
    for pred_index in range(len(pred_sentiments)):
        pred_target_text = TargetText.target_text_from_prediction(text=text, text_id=_id, tokenized_text=tokens, sequence_labels=pred_labels[pred_index])
        pred_spans.append(pred_target_text['spans'])
    
    # Create TargetTexts for each prediction with the gold annotations 
    # for those targets that have been correctly identified 
    for pred_index, pred_span in enumerate(pred_spans):
        pred_sentiment = pred_sentiments[pred_index]
        pred_label = pred_labels[pred_index]
        temp_pred_sentiment = []
        temp_pred_span = []
        temp_target_texts = []
        temp_gold_sentiment = []
        
        for span_index, span in enumerate(pred_span):
            if span in gold_spans:
                gold_index = gold_spans.index(span)
                temp_pred_sentiment.append(pred_sentiment[span_index])
                temp_pred_span.append(span)
                temp_gold_sentiment.append(gold_sentiments[gold_index])
                temp_target_texts.append(text[span.start: span.end])
        pred_target_text = TargetText(text_id=str(_id), text=text, tokenized_text=tokens, 
                                        targets=temp_target_texts,
                                        spans=temp_pred_span, target_sentiments=temp_gold_sentiment, 
                                        pred_sentiments=[temp_pred_sentiment])
        if len(result_structs) < (pred_index + 1):
            result_structs.append([])
        result_structs[pred_index].append(pred_target_text)
    return result_structs

def conll_to_custom_struct(conll_result_fp: Path, num_results: int = 5) -> Dict[str, Any]:
    '''
    Returns from the conll files the DS 1, 2, and 3 sample and sentence counts
    as well as their respective accuracies. Also it returns the multi STAC 
    sentence counts and accuracies. The return is like so:
    {'ds_sample_counts': ds_sample_counts, 'ds_sentence_counts': ds_sentence_counts, 
     'ds_accuracies': ds_accuracies, 'multi_stac_sentences': multi_sentence_count,
     'multi_stac_accuracies': multi_stac_accuracies}
    Where each value is a list representing the different model outputs e.g. if you 
    ran the model 5 times then this list will be of length 5. Then for the ds* 
    lists each of those model runs will also be a list of 3 where 3 represents 
    the different ds values 1, 2, and 3.
    '''
    result_structs = []
    with conll_result_fp.open('r') as conll_file:
        sentences = []
        _id = 0
        for line_index, line in enumerate(conll_file):
            if not sentences and not line.strip():
                continue
            elif line.strip():
                sentences.append(line)
            elif not line.strip() and sentences:
                # need to do somehting with the sentence by adding it to the collection
                result_structs = create_target_text(sentences, num_results, _id, 
                                                    result_structs)
                _id += 1
                sentences = []
            else:
                raise ValueError(f'There cannot be no sentences {sentences} and a line {line}')
        if sentences:
            result_structs = create_target_text(sentences, num_results, _id, 
                                                result_structs)
            _id += 1
            sentences = []
    from target_extraction.analysis.sentiment_error_analysis import distinct_sentiment, swap_and_reduce
    from target_extraction.analysis.sentiment_metrics import accuracy, strict_text_accuracy
    ds_sample_counts = []
    ds_sentence_counts = []
    ds_accuracies = []
    multi_stac_accuracies = []
    multi_sentence_count = []
    for result_struct in result_structs:
        ds_sample_count = [0,0,0]
        ds_sentence_count = [0,0,0]
        ds_accuracy = [0,0,0]

        result_struct = TargetTextCollection(result_struct)
        result_struct = distinct_sentiment(result_struct, separate_labels=True)
        unique_dss = result_struct.unique_distinct_sentiments('target_sentiments')
        for unique_ds in unique_dss:
            reduced_result_struct = swap_and_reduce(result_struct, f'distinct_sentiment_{unique_ds}', 
                                                    'target_sentiments', ['pred_sentiments'])
            ds_sentence_count[unique_ds-1] += len(reduced_result_struct)
            ds_sample_count[unique_ds-1] += reduced_result_struct.number_targets()
            ds_accuracy[unique_ds-1] += accuracy(reduced_result_struct, 'target_sentiments', 'pred_sentiments', average=False, array_scores=False)
        ds_accuracies.append(ds_accuracy)
        ds_sample_counts.append(ds_sample_count)
        ds_sentence_counts.append(ds_sentence_count)
        
        if 2 in unique_dss and 3 in unique_dss:
            reduce_keys = [f'distinct_sentiment_{ds_number}' 
                           for ds_number in [2,3]]
            reduced_result_struct = swap_and_reduce(result_struct, reduce_keys, 
                                                    'target_sentiments', ['pred_sentiments'])
            multi_sentence_count.append(len(reduced_result_struct))
            multi_stac_accuracies.append(strict_text_accuracy(reduced_result_struct, 'target_sentiments', 
                                                              'pred_sentiments', average=False, array_scores=False))
        elif 2 in unique_dss:
            reduced_result_struct = swap_and_reduce(result_struct, ['distinct_sentiment_2'], 
                                                    'target_sentiments', ['pred_sentiments'])
            multi_sentence_count.append(len(reduced_result_struct))
            multi_stac_accuracies.append(strict_text_accuracy(reduced_result_struct, 'target_sentiments', 
                                                              'pred_sentiments', average=False, array_scores=False))
        else:
            multi_sentence_count.append(0)
            multi_stac_accuracies.append(0.0)

    return {'ds_sample_counts': ds_sample_counts, 'ds_sentence_counts': ds_sentence_counts, 
            'ds_accuracies': ds_accuracies, 'multi_stac_sentences': multi_sentence_count,
            'multi_stac_accuracies': multi_stac_accuracies}

def parse_path(path_string: str) -> Path:
    _path = Path(path_string).resolve()
    return _path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=parse_path,
                        help='Directory containing all of the results')
    parser.add_argument("save_file", type=parse_path,
                        help='File to save the results too.')
    args = parser.parse_args()

    aux_tasks = ['conan_doyle', 'dr', 'lextag', 'sfu', 'sfu_spec', 'u_pos']
    mtl_models = [f'mtl${aux_task}' for aux_task in aux_tasks]
    model_mapper = {'mtl$conan_doyle': 'MTL (CD)', 'mtl$dr': 'MTL (DR)', 
                    'mtl$lextag': 'MTL (LEX)', 'mtl$sfu': 'MTL (SFU)', 
                    'mtl$sfu_spec': 'MTL (SPEC)', 'mtl$u_pos': 'MTL (UPOS)',
                    'stl': 'STL'}
    models = ['stl'] + mtl_models
    dataset_glove = ['laptop', 'restaurant', 'mpqa', 'MAMS']
    dataset_splits = ['dev.conll', 'test.conll']
    dataset_contextualized = [f'{dataset}_contextualized' for dataset in dataset_glove]
    datasets = dataset_glove + dataset_contextualized

    metric = []
    metric_name = []
    number_samples = []
    number_sentences = []
    embedding_names = []
    model_names = []
    data_split_names = []
    dataset_names = []
    model_run = []

    _dir = args.result_dir
    # model_name, embedding_name, dataset, split
    for model in models:
        if len(model.split('$')) == 2:
            model_dir = Path(_dir, *model.split('$')).resolve()
        else:
            model_dir = _dir / model
        for dataset in datasets:
            dataset_dir = model_dir / dataset
            for dataset_split in dataset_splits:
                result_fp = dataset_dir / dataset_split
                results = conll_to_custom_struct(result_fp)

                embedding = 'CWR' if 'contextualized' in dataset else 'GloVe'
                dataset_name = dataset.split('_')[0]
                split_name = dataset_split.split('.')[0]
                model_name = model_mapper[model]

                ds_samples = results['ds_sample_counts']
                ds_sentences = results['ds_sentence_counts']
                ds_accuracies = results['ds_accuracies']
                multi_stac_sentences = results['multi_stac_sentences']
                multi_stac_accuracies = results['multi_stac_accuracies']
                for model_number in range(5):
                    for ds_i in range(3):
                        number_samples.append(ds_samples[model_number][ds_i])
                        number_sentences.append(ds_sentences[model_number][ds_i])
                        metric.append(ds_accuracies[model_number][ds_i])
                        metric_name.append(f'DS{ds_i + 1}')
                        embedding_names.append(embedding)
                        dataset_names.append(dataset_name)
                        data_split_names.append(split_name)
                        model_names.append(model_name)
                        model_run.append(model_number)
                    
                    number_samples.append(multi_stac_sentences[model_number])
                    number_sentences.append(multi_stac_sentences[model_number])
                    metric.append(multi_stac_accuracies[model_number])
                    metric_name.append(f'STAC')

                    embedding_names.append(embedding)
                    dataset_names.append(dataset_name)
                    data_split_names.append(split_name)
                    model_names.append(model_name)
                    model_run.append(model_number)

    metric_df = {'Number Sentences': number_sentences, 'Number Samples': number_samples, 
                'Metric': metric, 'Embedding': embedding_names, 'Dataset': dataset_names,
                'Split': data_split_names, 'Model': model_names, 'Run': model_run,
                'Metric Name': metric_name}
    import json
    with args.save_file.open('w+') as save_fp:
        json.dump(metric_df, save_fp)