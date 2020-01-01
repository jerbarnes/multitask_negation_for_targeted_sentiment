from typing import List, Tuple, Optional, Dict, Set
import re
from collections import Counter, defaultdict
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def total_label_tokens(label_counter: Counter) -> int:
    '''
    :param label_counter: A Counter where the keys are the lengths of the 
                          label spans e.g. the label ['B_neg', 'I_neg'] would 
                          be 2 and if that is the only label span then the counter 
                          given here would be {2: 1}.
    :returns: Would be each key multiplied by each value. In the example case 
              this would return 2.
    '''
    total_count = 0
    for label_length, number_times_appeared in label_counter.items():
        total_count += label_length * number_times_appeared
    return total_count

def cue_or_scope(label_string: str, possible_labels: Set[str]) -> str:
    '''
    :param label_string: A String that is to be identified as either cue or 
                         scope an example label_string is `B_negcue` or 
                         `I_spec`
    :param possible_labels: A set of all possible labels that the `label_string`
                            is allowed to be.
    :returns: `cue` or `scope`
    :raises AssertionError: If the `label_string` is not in the `possible_labels`
                            set.
    '''
    assert_error = (f'The label string {label_string} has to be within the '
                    f'label set {possible_labels}')
    assert label_string in possible_labels, assert_error
    if re.search(r'cue$', label_string):
        return 'cue'
    else:
        return 'scope'

def assert_coding_scheme(labels: List[str], possible_labels: Set[str], 
                         sentence: Optional[str] = None) -> None:
    '''
    :param labels: A list of label strings e.g. [`B_negcue`, 'I_negcue']
    :param possible_labels: A set of all possible labels that the String in `labels`
                            is allowed to be.
    :param sentence: Will print the sentence the error occured in if given
                     here.
    :raises AssertionError: If the list of label are not in the BIO 
                            scheme where only B and I should be in the List 
                            of Strings where B is always first and only one 
                            then this will raise an error.
    :raises AssertionError: If the list contains more than one label e.g. 
                            ['B_neg', 'I_negcue']
    :raises AssertionError: If a String within `labels` is not in the `possible_labels`
                            set.
    '''
    error_sentence = ''
    if sentence:
        error_sentence = f' string this error occured in {sentence}'
    b_error = (f'The first label in a list of labels {labels} should have'
               f' have the `B` label encoding `{error_sentence}`')
    i_error = (f'All other labels in the list of labels {labels} should have'
               f' have the `I` label encoding after the first `{error_sentence}`')
    label_error = (f'All the labels within the {labels} should all be the'
                   f' same e.g. `negcue` `{error_sentence}`')

    assert 'B' == labels[0][0], b_error
    first_label = cue_or_scope(labels[0], possible_labels)
    if len(labels) == 1:
        return
    for label in labels[1:]:
        assert 'I' == label[0], i_error
        assert first_label == cue_or_scope(label, possible_labels), label_error

def update_label_list_count(label_list: List[str], possible_labels: Set[str],
                            label_list_counter: Dict[str, Dict[int, int]], 
                            sentence: Optional[str] = None) -> Dict[str, Counter]:
    '''
    :param label_list: A list of label strings e.g. [`B_negcue`, 'I_negcue']
    :param possible_labels: A set of all possible labels that the String in `label_list`
                            is allowed to be.
    :param label_list_counter: A defaultdict that contains a Counter where the 
                               outer dictionary keys are either `cue` or `scope`
                               and the Counter counts the length of the 
                               `label_list`
    :param sentence: Will print the sentence the error occured in if given
                     here.
    :returns: The updated `label_list_counter` where either the `cue` or `scope`
              will be updated with a new or incremented label list length.
    :raises ValueError: If the label in all of the labels from the `label_list`
                        argument is not `cue` or `scope`
    :raises AssertionError: If a String within `label_list` is not in the `possible_labels`
                            set.
    '''
    assert_coding_scheme(label_list, possible_labels, sentence)
    label_list_length = len(label_list)
    if cue_or_scope(label_list[0], possible_labels) == 'cue':
        label_list_counter['cue'].update([label_list_length])
    elif cue_or_scope(label_list[0], possible_labels) == 'scope':
        label_list_counter['scope'].update([label_list_length])
    else:
        label_error = ('The label in the label list is neither `cue` or `scope`'
                       f' all the `label_list` given is {label_list}')
        raise ValueError(label_error)
    return label_list_counter

def get_label_length_counter(data_path: Path, dataset_name: str, label_type: str
                             ) -> Tuple[int, int, Counter]:
    '''
    :param data_path: The path to the negation or speculation dataset to parse
    :param dataset_name: The name of the dataset
    :param label_type: Either negation or speculation labels
    :returns: A tuple of 1. Number of label sentences in the dataset, 
              2. Number of sentences in the dataset, and 
              3. A python Counter where the 
              keys are the length of the label span and the value is the number 
              of times that label span length has occured in the dataset given.
    '''
    possible_classes = set()
    if dataset_name == 'sfu':
        if label_type == 'negation':
            possible_classes = {'O', 'I_neg', 'B_neg', 
                                'B_negcue', 'I_negcue'}
        else:
            possible_classes = {'O', 'I_spec', 'B_spec', 'B_speccue', 
                                'I_speccue'}
    else:
        possible_classes = {'O', 'B_scope', 'I_scope', 'B_cue', 'I_cue'}
    
    number_of_sentences = 0
    contain_label_in_sentence = []
    in_sentence = False
    current_sentence = ''
    current_label: List[str] = []
    label_lengths = defaultdict(lambda: Counter())
    
    with data_path.open('r') as lines:
        sentence_contain_label = False
        for line_index, line in enumerate(lines):
            if re.search(r'^#token.*', line) or re.search(r'# document - .', line):
                continue
            elif re.search(r'^# .*', line):
                in_sentence = True
                if not sentence_contain_label:
                    contain_label_in_sentence.append(0)
                sentence_contain_label = False
                current_sentence = line
                number_of_sentences += 1
                if len(current_label):
                    label_lengths = update_label_list_count(current_label, possible_classes,
                                                            label_lengths, current_sentence)
                    current_label = []
            # When blank line then it is at the end of the sentence
            elif not line.strip() or line.strip() == '#token\tnegation_scope':
                in_sentence = False
                if len(current_label):
                    label_lengths = update_label_list_count(current_label, possible_classes,
                                                            label_lengths, current_sentence)
                    current_label = []
            elif in_sentence:
                line_splits = line.split('\t')
                the_label = None
                if dataset_name == 'sfu':
                    assert_error = ('Format for SFU is 3 columns not '
                                    f'{line_splits}, line index {line_index}')
                    assert 3 == len(line_splits), assert_error
                    line_splits[2] = line_splits[2].strip()
                    if label_type == 'negation':
                        the_label = line_splits[1]
                    else:
                        the_label = line_splits[2]
                else:
                    assert_error = ('Format for Conandoyle is 2 columns not '
                                    f'{line_splits}, line index {line_index}')
                    assert 2 == len(line_splits), assert_error
                    line_splits[1] = line_splits[1].strip()
                    the_label = line_splits[1]

                label_encoding = the_label[0]
                if len(current_label):
                    if label_encoding == 'B':
                        label_lengths = update_label_list_count(current_label, possible_classes,
                                                            label_lengths, current_sentence)
                        current_label = []
                        current_label.append(the_label)
                    elif label_encoding == 'I':
                        current_label.append(the_label)
                else:
                    if label_encoding == 'B':
                        if not sentence_contain_label:
                            sentence_contain_label = True
                            contain_label_in_sentence.append(1)
                        current_label.append(the_label)
                    elif label_encoding == 'I':
                        raise ValueError('This does not conform to the label ')
            elif re.search(r'^# document.*', line):
                if len(current_label):
                    label_lengths = update_label_list_count(current_label, possible_classes,
                                                            label_lengths, current_sentence)
                    current_label = []
                in_sentence = False
            else:
                value_error = (f'This line should not exist: {line}, '
                               f'line index {line_index}')
                raise ValueError(value_error)
    if not sentence_contain_label:
        contain_label_in_sentence.append(0)
    number_negated_sentences = sum(contain_label_in_sentence)
    return number_negated_sentences, number_of_sentences, label_lengths

if __name__ == '__main__':
    label_type_help = ("Label to get the statistics for. Either `negation` "
                       "or `speculation`")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=parse_path, 
                        help="Path to the negation or speculation dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset", 
                        choices=['sfu', 'conandoyle'])
    parser.add_argument("label_type", type=str, 
                        choices=["negation", "speculation"],
                        help=label_type_help)
    args = parser.parse_args()

    output = get_label_length_counter(args.data_path, args.dataset_name, 
                                      args.label_type)
    number_label_sentences, number_sentences, label_span_counter = output

    print(f'Dataset: {args.dataset_name}')
    print(f'Label: {args.label_type}')
    print(f'Number of cue token labels: {total_label_tokens(label_span_counter["cue"])}')
    print(f'Number of cues: {sum(label_span_counter["cue"].values())}')
    print(f'Number of scope token labels: {total_label_tokens(label_span_counter["scope"])}')
    print(f'Number of scopes: {sum(label_span_counter["scope"].values())}')
    print(f'Number of sentences that contain {args.label_type}: {number_label_sentences}')
    print(f'Number of sentences: {number_sentences}')
