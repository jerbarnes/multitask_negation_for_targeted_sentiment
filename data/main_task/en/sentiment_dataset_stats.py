import argparse
from collections import Counter
from os import read
from typing import Tuple, List, Dict, Union
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def read_conll(file: str) -> Tuple[List[List[str]], 
                                   List[List[str]], 
                                   List[List[str]]]:
    '''
    :returns: A tuple of 1. The list of sentences represented as a list of tokens.
              2. A list of targets whereby each list is associated with the same 
              indexed sentence and each target is represented as a list of tokens.
              3. A list of sentiment polarities that are associated to the targets.

    Example return: ([['this', 'was', 'good', 'today'], 
                      ['yesterday', 'was', 'bad', 'but', 'tomorrow', 'will', 'be', 'worse']
                     ], [['today'], ['yesterday', 'tomorrow']], 
                     [['pos'], ['neg', 'neg']])
    '''
    sents, targets, labels = [], [], []
    sent, targ, label = [], [], []
    current_targ = []
    for line in open(file):
        if line.strip() == "":
            if current_targ != []:
                targ.append(current_targ)
            if sent != []:
                sents.append(sent)
                targets.append(targ)
                labels.append(label)
            sent, targ, label = [], [], []
        else:
            tok, l = line.strip().split()
            sent.append(tok)
            if l[0] == "B":
                bio, pol = l.split("-")
                current_targ.append(tok)
                label.append(pol)
            elif l[0] == "I":
                current_targ.append(tok)
            elif l[0] == "L":
                current_targ.append(tok)
                targ.append(current_targ)
                current_targ = []
            elif l[0] == "U":
                bio, pol = l.split("-")
                current_targ.append(tok)
                label.append(pol)
                targ.append(current_targ)
                current_targ = []
    return sents, targets, labels

def create_statistics(formatted_conll_data: Tuple[List[List[str]], 
                                                  List[List[str]], 
                                                  List[List[str]]]) -> Dict[str, Union[int, float]]:
    sents, targets, labels = formatted_conll_data
    stats_dict = {}
    
    # Number of sentences
    num_sentences = len(sents)
    stats_dict['sents.'] = num_sentences
    
    # Number of Targets
    num_targets = len([i for j in targets for i in j])
    stats_dict['targs.'] = num_targets
    
    # Average Length of Targets
    lengths = [len(i) for j in targets for i in j]
    ave = np.mean(lengths)
    stats_dict['len.'] = round(ave, 1)

    # Number of sents with multiple conflicting polarities
    multi_polarity_count = 0
    for label_sent in labels:
        if len(set(label_sent)) > 1:
            multi_polarity_count += 1
    stats_dict['mult.'] = multi_polarity_count

    # Sentiment label percentages
    label_mapper = {'negative': 'NEG', 'neutral': 'NEU', 'positive': 'POS', 'both': 'BOTH'}
    label_mapper = {'NEG': 'NEG', 'NEU': 'NEU', 'POS': 'POS', 'BOTH': 'BOTH', **label_mapper}
    sent_label_count = Counter()
    for sent_label in labels:
        sent_label_count.update(sent_label)
    for label, count in sent_label_count.items():
        percentage = round(float(count) / float(num_targets), 4) * 100
        stats_dict[label_mapper[label]] = percentage
    if 'BOTH' not in stats_dict:
        stats_dict['BOTH'] = 0.0

    return stats_dict

def dataset_data_stats(dataset_split : Dict[str, Dict[str, str]], 
                       split_names: List[str]) -> pd.DataFrame:
    '''
    :param dataset_split: A dict of keys containing dataset name, and a value of 
                          a dict with a key of split name and the value being 
                          a file path as a string that contains that dataset 
                          split.
    :param split_names: The list of possible split names. This is required as 
                        some datasets do not contain some splits.
    :returns: A pandas dataframe with the index being the dataset name and the 
              columns to be a hierarchical index of split name as the top level 
              and the lower levels [sents., targs., len., mult., POS, NEU, NEG]
    '''
    all_dataset_stats: List[Dict[str, Union[int, float, str]]] = []
    for dataset_name, split_name_fp in dataset_split.items():
        for split_name in split_names:
            if split_name not in split_name_fp:
                split_stats: Dict[str, Union[int, float, str]] = {}
                split_stats['split'] = split_name
                split_stats['dataset'] = dataset_name
                split_stats['sents.'] = 0
                split_stats['targs.'] = 0
                split_stats['mult.'] = 0
                split_stats['len.'] = 0.0
                split_stats['POS'] = 0.0
                split_stats['NEG'] = 0.0
                split_stats['NEU'] = 0.0
                split_stats['BOTH'] = 0.0
                all_dataset_stats.append(split_stats)
            else:
                fp = split_name_fp[split_name]
                stats = create_statistics(read_conll(fp))
                split_stats: Dict[str, Union[int, float, str]] = {**stats}
                split_stats['split'] = split_name
                split_stats['dataset'] = dataset_name
                all_dataset_stats.append(split_stats)
    dataset_stats_df = pd.DataFrame(all_dataset_stats)
    df_columns: List[str] = ['sents.', 'targs.', 'len.', 'mult.', 'POS', 'NEU', 'NEG', 'BOTH']
    split_name_dfs = []
    for split_name in split_names:
        split_name_df = dataset_stats_df[dataset_stats_df['split']==split_name]
        split_name_df = split_name_df.drop('split', axis=1)
        split_name_df = split_name_df.set_index('dataset')
        split_name_df.columns = pd.MultiIndex.from_tuples(list(product([split_name], df_columns)))
        split_name_dfs.append(split_name_df)
    return pd.concat(split_name_dfs, axis=1)

if __name__ == '__main__':
    description = '''
    Outputs the dataset statistics for the main and challenge datasets in 
    various formats. By default is outputted as a pandas Dataframe. 
    To combine the main and challenge datasets together run with both 
    flags at the same time e.g. --main-datasets --challenge-datasets
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--main-datasets', action='store_true', 
                        help='Output dataset statistics for the main datasets used in the experiments.')
    parser.add_argument('--challenge-datasets', action='store_true', 
                        help='Output dataset statistics for the challenge datasets used in the experiments.')
    parser.add_argument('--to-html', action='store_true', help='Output is HTML formatted')
    parser.add_argument('--to-markdown', action='store_true', help='Output is in markdown')
    parser.add_argument('--to-latex', action='store_true', help='Output is in latex')

    args = parser.parse_args()

    if not args.main_datasets and not args.challenge_datasets:
        error_msg = '''
        Either the --main-datasets and/or --challenge-datasets have to be flagged.
        '''
        raise ValueError(error_msg)
    
    data_dir = Path(__file__, '..').resolve()
    datasets: List[pd.DataFrame] = []
    dataset_columns = None
    if args.main_datasets:
        split_names = ['train', 'dev', 'test']
        dataset_names = ['laptop', 'restaurant', 'MAMS', 'mpqa']
        dataset_split_fps = {}
        for dataset_name in dataset_names:
            dataset_dir = Path(data_dir, dataset_name)
            split_fps = {split_name: dataset_dir / f'{split_name}.conll' 
                         for split_name in split_names}
            dataset_split_fps[dataset_name] = split_fps
        temp_dataset = dataset_data_stats(dataset_split_fps, split_names)
        dataset_columns = temp_dataset.columns
        datasets.append(temp_dataset)


    if args.challenge_datasets:
        split_names = ['dev', 'test']
        dataset_names = ['laptop', 'restaurant']
        challenge_types = ['neg', 'spec']
        dataset_split_fps = {}
        for dataset_name in dataset_names:
            dataset_dir = Path(data_dir, dataset_name)
            for challenge_type in challenge_types:
                split_fps = {split_name: dataset_dir / f'{split_name}_{challenge_type}_only.conll' 
                             for split_name in split_names}
                dataset_split_fps[f'{dataset_name}_{challenge_type}'] = split_fps
        temp_dataset = dataset_data_stats(dataset_split_fps, split_names)
        if not args.main_datasets:
            dataset_columns = temp_dataset.columns
        datasets.append(temp_dataset)
    dataset_statistics: pd.DataFrame = pd.concat(datasets, axis=0)
    dataset_statistics = dataset_statistics[dataset_columns]
    
    to_html = 1 if args.to_html else 0
    to_latex = 1 if args.to_latex else 0
    to_markdown = 1 if args.to_markdown else 0
    if (to_html + to_latex + to_markdown) > 1:
        error_msg = '''
        Can only have one of the following flags:
        1. --to-html, 2. --to-markdown, 3. --to-latex
        '''
        raise ValueError(error_msg)
    elif to_html:
        print(dataset_statistics.to_html())
    elif to_markdown:
        print(dataset_statistics.to_markdown())
    elif to_latex:
        print(dataset_statistics.to_latex())
    else:
        print(dataset_statistics)