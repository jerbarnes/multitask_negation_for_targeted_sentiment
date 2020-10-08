import argparse
from typing import List, Tuple, Dict, Union
from collections import Counter
from itertools import product

import pandas as pd
import numpy as np

def read_conll(file: str) -> Tuple[List[List[str]], List[List[str]], 
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="laptop/train.conll")

    args = parser.parse_args()

    sents, targets, labels = read_conll(args.file)

    print()
    # Number of Targets
    num_targets = len([i for j in targets for i in j])
    print("Number of Targets: {0}".format(num_targets))

    # Average Length of Targets
    lengths = [len(i) for j in targets for i in j]
    ave = np.mean(lengths)
    print("Avg. Target Length: {0:.1f}".format(ave))

    # Number of sents with multiple conflicting polarities
    count = 0
    for sent in labels:
        if len(set(sent)) > 1:
            count += 1
    print("Number of Multiple Polarities: {0}".format(count))
    print()
