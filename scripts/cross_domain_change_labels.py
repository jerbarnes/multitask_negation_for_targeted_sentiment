from typing import Dict, Iterator, Any, Tuple
import copy
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from multitask_negation_target.analysis import evaluation_metrics

def conll_label_re_mapping(conll_fp: Path, new_conll_fp: Path, 
                           mapper: Dict[str, str]) -> None:
    
    with conll_fp.open('r') as conll_file:
        with new_conll_fp.open('w+') as new_conll_file:
            for line in conll_file:
                if line.split():
                    values = line.split()
                    new_values = copy.deepcopy(values)
                    for index, value in enumerate(values):
                        # The first value is always the token
                        if index == 0:
                            continue
                        if value.lower() == 'o':
                            continue
                        bio, label = value.split('-')
                        if label in mapper:
                            label = mapper[label]
                        new_value = f'{bio}-{label}'
                        new_values[index] = new_value
                    new_line = ' '.join(new_values)
                    new_conll_file.write(f'{new_line}\n')
                else:
                    new_conll_file.write(line)

def get_generate_fps() -> Iterator[Tuple[str, str, str, str, Path]]:
    

    base_dir = Path(__file__, '..', '..','data', 'results', 'en', 'cross_domain')
    model_names = ['STL', 'MTL (UPOS)', 'MTL (DR)', 'MTL (LEX)', 'MTL (CD)', 
                   'MTL (SFU)', 'MTL (SPEC)']
    dataset_names = ['mpqa', 'Restaurant', 'Laptop', 'MAMS']
    split_names = ['Validation', 'Test']
    for model_name in model_names:
        if model_name == 'STL':
            model_dir = base_dir / 'stl'
        elif model_name == 'MTL (CD)':
            model_dir = base_dir / 'mtl' / 'conan_doyle'
        elif model_name == 'MTL (SFU)':
            model_dir = base_dir / 'mtl' / 'sfu'
        elif model_name == 'MTL (UPOS)':
            model_dir = base_dir / 'mtl' / 'u_pos'
        elif model_name == 'MTL (DR)':
            model_dir = base_dir / 'mtl' / 'dr'
        elif model_name == 'MTL (LEX)':
            model_dir = base_dir / 'mtl' / 'lextag'
        elif model_name == 'MTL (SPEC)':
            model_dir = base_dir / 'mtl' / 'sfu_spec'
        for trained_dataset_name in dataset_names:
            for tested_dataset_name in dataset_names:
                if trained_dataset_name == tested_dataset_name:
                    continue
                for split_name in split_names:
                    if split_name == 'Test':
                        if trained_dataset_name == 'MAMS':
                            result_fp = model_dir / f'{trained_dataset_name}' / f'{tested_dataset_name.lower()}' / 'test.conll'
                        else:
                            result_fp = model_dir / f'{trained_dataset_name.lower()}' / f'{tested_dataset_name.lower()}' / 'test.conll'
                        if tested_dataset_name == 'MAMS':
                            result_fp = model_dir / f'{trained_dataset_name.lower()}' / f'{tested_dataset_name}' / 'test.conll'
                    elif split_name == 'Validation':
                        if trained_dataset_name == 'MAMS':
                            result_fp = model_dir / f'{trained_dataset_name}' / f'{tested_dataset_name.lower()}' / 'dev.conll'
                        else:
                            result_fp = model_dir / f'{trained_dataset_name.lower()}' / f'{tested_dataset_name.lower()}' / 'dev.conll'
                        if tested_dataset_name == 'MAMS':
                            result_fp = model_dir / f'{trained_dataset_name.lower()}' / f'{tested_dataset_name}' / 'dev.conll'
                    yield (model_name, trained_dataset_name, 
                           tested_dataset_name, split_name, 
                           result_fp.resolve())

def generate_results(model_name: str, trained_dataset_name: str, 
                     tested_dataset_name: str, 
                     split_name: str, fp: Path,
                     results_df_dict: Dict[str, Any]) -> Dict[str, Any]:
    
    for run_number in range(5):
        f1_a = evaluation_metrics.span_f1(fp, run_number=run_number, 
                                        ignore_sentiment=True)
        f1_a_recall, f1_a_precision, f1_a = f1_a
        f1_i = evaluation_metrics.span_f1(fp, run_number=run_number, 
                                        ignore_sentiment=False)
        f1_i_recall, f1_i_precision, f1_i = f1_i
        pos_filter_name = 'POS'
        neu_filter_name = 'NEU'
        neg_filter_name = 'NEG'

        f1_i_pos = evaluation_metrics.span_f1(fp, run_number=run_number, 
                                            ignore_sentiment=False, 
                                            filter_by_sentiment=pos_filter_name)
        f1_i_pos_recall, f1_i_pos_precision, f1_i_pos = f1_i_pos
        f1_i_neu = evaluation_metrics.span_f1(fp, run_number=run_number, 
                                            ignore_sentiment=False, 
                                            filter_by_sentiment=neu_filter_name)
        f1_i_neu_recall, f1_i_neu_precision, f1_i_neu = f1_i_neu
        f1_i_neg = evaluation_metrics.span_f1(fp, run_number=run_number, 
                                            ignore_sentiment=False, 
                                            filter_by_sentiment=neg_filter_name) 
        f1_i_neg_recall, f1_i_neg_precision, f1_i_neg = f1_i_neg
        f1_s = evaluation_metrics.span_label_metric(fp, run_number=run_number, 
                                                    metric_func=f1_score, 
                                                    average='macro')
        acc_s = evaluation_metrics.span_label_metric(fp, run_number=run_number,
                                                    metric_func=accuracy_score)
        results_df_dict['Model'].append(model_name)
        results_df_dict['Trained Dataset'].append(trained_dataset_name)
        results_df_dict['Tested Dataset'].append(tested_dataset_name)
        results_df_dict['Split'].append(split_name)
        results_df_dict['Run'].append(run_number)
        results_df_dict['F1-a'].append(f1_a)
        results_df_dict['F1-a-R'].append(f1_a_recall)
        results_df_dict['F1-a-P'].append(f1_a_precision)
        results_df_dict['F1-i'].append(f1_i)
        results_df_dict['F1-i-R'].append(f1_i_recall)
        results_df_dict['F1-i-P'].append(f1_i_precision)
        results_df_dict['F1-i-pos'].append(f1_i_pos)
        results_df_dict['F1-i-pos-R'].append(f1_i_pos_recall)
        results_df_dict['F1-i-pos-P'].append(f1_i_pos_precision)
        results_df_dict['F1-i-neg'].append(f1_i_neg)
        results_df_dict['F1-i-neg-R'].append(f1_i_neg_recall)
        results_df_dict['F1-i-neg-P'].append(f1_i_neg_precision)
        results_df_dict['F1-i-neu'].append(f1_i_neu)
        results_df_dict['F1-i-neu-R'].append(f1_i_neu_recall)
        results_df_dict['F1-i-neu-P'].append(f1_i_neu_precision)
        results_df_dict['F1-s'].append(f1_s)
        results_df_dict['Acc-s'].append(acc_s)
    return results_df_dict



if __name__ == '__main__':
    change_labels_help = ('Whether the script should change the labels so '
                          'that they are all POS, NEG, and NEU')
    import argparse
    import tempfile
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("change_labels", type=str,
                        help=change_labels_help)
    args = parser.parse_args()
    change_labels = args.change_labels
    if change_labels == 'False':
        change_labels = False
    else:
        change_labels = True
    if change_labels:
        mapper = {'positive': 'POS','negative': 'NEG','neutral': 'NEU'}
        for values in get_generate_fps():
            fp = values[4]
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir, 'temp_file')
                conll_label_re_mapping(fp, temp_file, mapper=mapper)
                shutil.move(str(temp_file), str(fp))
    else:
        results_df_dict = defaultdict(list)
        results_fp = Path(__file__, '..', '..','data', 'results', 'en', 
                          'cross_domain', 'results.json').resolve()
        for values in get_generate_fps():
            model_name, trained_dataset_name, tested_dataset_name, split_name, fp = values
            results_df_dict = generate_results(model_name, trained_dataset_name, 
                                               tested_dataset_name, split_name, fp, 
                                               results_df_dict)
        with results_fp.open('w+') as results_file:
            json.dump(results_df_dict, results_file)