from typing import List, Tuple
import re
import random
from pathlib import Path

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def choose_file_write_lines(file_paths: List[Path], file_probabilities: List[float], 
                            lines: List[str]) -> None:
    '''
    :param file_paths: A list of file paths
    :param file_probabilities: The probability of choosing the associated file 
                               path to write the lines to. 
    :param lines: The lines to write to the selected file based on `file_probabilities`
    '''
    data_fp = random.choices(file_paths, weights=file_probabilities)[0]
    with data_fp.open('a+') as data_file:
        for line in lines:
            line = f'{line.strip()}\n'
            data_file.write(line)
        data_file.write('\n')

def sfu_splitter(train_dev_test_probability: Tuple[float, float, float],
                 train_dev_test_file_paths: Tuple[Path, Path, Path],
                 sfu_data_file_path: Path) -> None:
    '''
    :param train_dev_test_probability: The amount to split the SFU dataset into 
                                       train, dev, and test splits e.g. 
                                       (0.7,0.2,0.1) would split the dataset 
                                       into 70%, 20%, and 10% train, dev, and 
                                       test data from the SFU data that comes 
                                       from `sfu_data_file_path` argument.
    :param train_dev_test_file_paths: File path to store the train, dev, and 
                                      test data to.
    :param sfu_data_file_path: File Path to the SFU data.
    :raises FileExistsError: If any of the `train_dev_test_file_paths` exist
    '''
    train_probability = train_dev_test_probability[0]
    dev_probability = train_dev_test_probability[1]
    test_probability = train_dev_test_probability[2]

    train_path = train_dev_test_file_paths[0]
    dev_path = train_dev_test_file_paths[1]
    test_path = train_dev_test_file_paths[2]

    data_split_paths = [train_path, dev_path, test_path]
    for data_split_path in data_split_paths:
        if data_split_path.exists():
            file_exist_error = ('None of the data split files can exist and '
                                f'this file does exist {data_split_path}.'
                                ' To re-run this without error please delete '
                                'this file or choose a different File Path')
            raise FileExistsError(file_exist_error)
    weights = [train_probability, dev_probability, test_probability]

    in_sentence = False
    current_domain = None
    with sfu_data_file_path.open('r') as lines:
        sentence_lines: List[str] = []
        for line in lines:
            if re.search(r'^# document - .*', line):
                domain  = re.search(r'\s\w+/', line)
                if domain:
                    domain = domain[0][1:-1]
                else:
                    raise ValueError(f'This line should contain a domain name {line}')
                current_domain = f'# domain - {domain}'
            elif re.search(r'^# .*', line):
                if sentence_lines:
                    # choose which dataset split to write the sentence too
                    choose_file_write_lines(data_split_paths, weights, sentence_lines)
                    sentence_lines = []
                in_sentence = True
                sentence_lines.append(current_domain)
                sentence_lines.append(line)
            elif not line.strip():
                continue
            else:
                sentence_lines.append(line)
        if sentence_lines:
            # choose which dataset split to write the sentence too
            choose_file_write_lines(data_split_paths, weights, sentence_lines)
            sentence_lines = []



if __name__ == '__main__':
    set_random_seed_help = ("Whether to set the random seed thus making the "
                            "dataset split generation reproducible")
    sfu_split_dir_help = ("Directory to save the SFU train, dev, and test "
                          "split too. The files will be called SFU_train.conll,"
                          " SFU_dev.conll, and SFU_test.conll within this directory")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_split_probability", type=float, 
                        help='The fraction of data to give to the train split')
    parser.add_argument("dev_split_probability", type=float, 
                        help='The fraction of data to give to the dev split')
    parser.add_argument("test_split_probability", type=float, 
                        help='The fraction of data to give to the test split')
    parser.add_argument("sfu_split_dir", type=parse_path, 
                        help=sfu_split_dir_help)
    parser.add_argument("sfu_data_fp", type=parse_path, 
                        help='File Path to the SFU data')
    parser.add_argument("--set_random_seed", action="store_true", help=set_random_seed_help)
    args = parser.parse_args()
    
    if args.set_random_seed:
        random.seed(42)
    
    sfu_data_dir = args.sfu_split_dir
    sfu_data_dir.mkdir(parents=True, exist_ok=True)

    train_fp = Path(sfu_data_dir, 'SFU_train.conll')
    dev_fp = Path(sfu_data_dir, 'SFU_dev.conll')
    test_fp = Path(sfu_data_dir, 'SFU_test.conll')
    probabilities = (args.train_split_probability, args.dev_split_probability,
                     args.test_split_probability)


    sfu_splitter(train_dev_test_probability=probabilities,
                 train_dev_test_file_paths=(train_fp, dev_fp, test_fp),
                 sfu_data_file_path=args.sfu_data_fp)