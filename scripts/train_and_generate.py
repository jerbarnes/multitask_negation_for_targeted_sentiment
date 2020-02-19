from pathlib import Path
from typing import Dict, Any, Iterable, List
import random
import json
import tempfile
import shutil

from allennlp.commands.train import train_model_from_file
from allennlp.common import from_params, Params
from allennlp.data import DatasetReader, Token
import target_extraction

if __name__ == '__main__':
    import sys
    from pathlib import Path
    package_path = str(Path(__file__, '..', '..').resolve())
    sys.path.insert(0, package_path)

import multitask_negation_target

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def create_input_sentences(conll_fp: Path) -> Iterable[List[Token]]:
    with conll_fp.open('r') as conll_file:
        tokens: List[Token] = []
        for line in conll_file:
            if line.strip():
                tokens.append(Token(line.split()[0]))
            else:
                yield(tokens)
                tokens = []
        if tokens:
            yield tokens

def write_tags_to_file(conll_fp: Path, predicted_tags: List[List[str]]) -> None:
    conll_line = 0
    conll_token_count = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fp = Path(temp_dir, 'temp_file.conll')
        with conll_fp.open('r') as conll_file:
            with temp_fp.open('w+') as temp_file:
                for line in conll_file:
                    line = line.strip()
                    if line:
                        pred_tag = predicted_tags[conll_line][conll_token_count]
                        line = f'{line} {pred_tag}\n'
                        temp_file.write(line)
                        conll_token_count += 1
                    else:
                        temp_file.write('\n')
                        conll_line += 1
                        conll_token_count = 0
        with temp_fp.open('r') as temp_file:
            with conll_fp.open('w+') as conll_file:
                for line in temp_file:
                    conll_file.write(line)

def count_number_runs(conll_fp: Path) -> int:
    with conll_fp.open('r') as conll_file:
        for line in conll_file:
            line = line.strip()
            if line:
                number_coulmns = len(line.split())
                # the first two columns represent token, gold label and the rest
                # are predictions
                number_runs = number_coulmns - 2
                return number_runs

if __name__ == '__main__':
    mtl_help = 'Whether or not the model being ran is a Multi Task Learning model'
    save_model_dir_help = ('Directory to save all models, there will be one '
                           'directory for each run containing the model')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_fp", type=parse_path,
                        help='File Path to model configuration')
    parser.add_argument('test_fp', type=parse_path, help='Test file path')
    parser.add_argument('val_fp', type=parse_path, help='Validation file path')
    parser.add_argument('save_results_dir', type=parse_path, 
                        help='Directory to save the test and validation prediction results')
    parser.add_argument('number_runs', type=int, help='Number of model runs')
    parser.add_argument('save_model_dir', type=parse_path, help=save_model_dir_help)
    parser.add_argument('--mtl', action='store_true', 
                        help=mtl_help)
    parser.add_argument('--aux_name', default='negation', type=str)
    args = parser.parse_args()
    save_results_dir = args.save_results_dir
    # create save directory if it does not exist
    if not save_results_dir.exists():
        save_results_dir.mkdir(parents=True, exist_ok=True)
    test_save_fp = Path(save_results_dir, 'test.conll')
    dev_save_fp = Path(save_results_dir, 'dev.conll')
    save_fps = [test_save_fp, dev_save_fp]
    original_fps = [args.test_fp, args.val_fp]
    # Transfer the tokens and gold labels if the file does not exist
    for save_fp, original_fp in zip(save_fps, original_fps):
        if save_fp.exists():
            continue
        with save_fp.open('w+') as save_file:
            with original_fp.open('r') as original_file:
                for line in original_file:
                    save_file.write(line)
    # Find out the number of runs done so far
    runs_already_done = count_number_runs(dev_save_fp)
    print(f'Number of runs already done {runs_already_done}')
    runs_to_do = args.number_runs - runs_already_done
    if runs_to_do < 1:
        raise ValueError('All of the model runs have been performed. '
                         f'Number of runs completed {runs_already_done}')
    random.seed(a=None)
    print(f'Number of runs to perform {runs_to_do}')
    model_dir = args.save_model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for run_number in range(runs_already_done, args.number_runs):
        print(f'Run number {run_number}')
        np_seed = random.randint(0, 9999)
        py_seed = random.randint(0, 9999)
        rand_seed = random.randint(0, 9999)
        overrides_string = {"numpy_seed": np_seed, "pytorch_seed": py_seed, 
                            "random_seed": rand_seed}
        overrides_string = json.dumps(overrides_string)
        model_save_fp = Path(model_dir, f'model_{run_number}.tar.gz')
        if model_save_fp.exists():
            raise FileExistsError(f'The model run file {model_save_fp} '
                                  'cannot already exist for you to save a model to it')
        aux_model_save_fp = None
        if args.mtl:
            aux_model_save_fp = Path(model_dir, f'task_{args.aux_name}_model_{run_number}.tar.gz')
            if aux_model_save_fp.exists():
                raise FileExistsError(f'The model run file {aux_model_save_fp} '
                                      'cannot already exist for you to save a model to it')
        with tempfile.TemporaryDirectory() as temp_data_dir:
            results = train_model_from_file(args.model_config_fp, 
                                            serialization_dir=temp_data_dir,
                                            overrides=overrides_string)
            params = Params.from_file(str(args.model_config_fp))
            if 'dataset_reader' not in params:
                reader = DatasetReader.from_params(params['task_sentiment']['dataset_reader'])
            else:
                reader = DatasetReader.from_params(params['dataset_reader'])
            results.eval()

            for save_fp, original_fp in zip(save_fps, original_fps):
                predicted_tags: List[List[str]] = []
                for pred_tokens in create_input_sentences(original_fp):
                    instance = reader.text_to_instance(pred_tokens)
                    pred_output = results.forward_on_instance(instance)
                    tags = pred_output['tags']
                    assert_err = (f'Number of predicted tags {len(pred_tokens)} should '
                                  f'match the number of tokens being predicted {len(tags)}')
                    assert len(tags) == len(pred_tokens), assert_err
                    predicted_tags.append(tags)
                write_tags_to_file(save_fp, predicted_tags)
                print('done')
            temp_save_model_fp = Path(temp_data_dir, 'model.tar.gz')
            if not Path(temp_data_dir).exists():
                raise FileNotFoundError('The model was not saved in the temp '
                                        f'directory {temp_save_model_fp}')
            if args.mtl:
                temp_aux_save_model_fp = Path(temp_data_dir, 'task_negation_model.tar.gz')
                if not Path(temp_aux_save_model_fp).exists():
                    raise FileNotFoundError('The model was not saved in the temp '
                                            f'directory {temp_aux_save_model_fp}')
                shutil.move(temp_aux_save_model_fp, aux_model_save_fp)
            shutil.move(temp_save_model_fp, model_save_fp)