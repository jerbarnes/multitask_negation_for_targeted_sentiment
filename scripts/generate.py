from pathlib import Path
from typing import Iterable, List
import tempfile
import re

from allennlp.common import from_params
from allennlp.data import DatasetReader, Token
from allennlp.models.archival import load_archive
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
                if tokens:
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
                was_empty_line = True
                for line in conll_file:
                    line = line.strip()
                    if line:
                        pred_tag = predicted_tags[conll_line][conll_token_count]
                        line = f'{line} {pred_tag}\n'
                        temp_file.write(line)
                        conll_token_count += 1
                        was_empty_line = False
                    elif was_empty_line:
                        temp_file.write('\n')
                    else:
                        temp_file.write('\n')
                        conll_line += 1
                        conll_token_count = 0
                        was_empty_line = True
        with temp_fp.open('r') as temp_file:
            with conll_fp.open('w+') as conll_file:
                for line in temp_file:
                    conll_file.write(line)

if __name__ == '__main__':
    mtl_help = 'Whether or not the model being ran is a Multi Task Learning model'
    save_model_dir_help = ('Directory to save all models, there will be one '
                           'directory for each run containing the model')
    model_dir_help = ('File Path to model the saved models where each model '
                      'in the directory is trained on a different random seed')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=parse_path,
                        help=model_dir_help)
    parser.add_argument('evaluation_fp', type=parse_path, 
                        help='Data to evaluate on')
    parser.add_argument('save_results_dir', type=parse_path, 
                        help='Directory to save the evaluation prediction results')
    parser.add_argument('--cuda', action="store_true", 
                        help='Whether the model uses CUDA else CPU')
    args = parser.parse_args()
    save_results_dir = args.save_results_dir
    # create save directory if it does not exist
    if not save_results_dir.exists():
        save_results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_fp = args.evaluation_fp
    save_fp = Path(save_results_dir, evaluation_fp.name)
    
    cuda = 0 if args.cuda else -1

    # Transfer the tokens and gold labels 
    with save_fp.open('w+') as save_file:
        with evaluation_fp.open('r') as evaluation_file:
            for line in evaluation_file:
                save_file.write(line)

    model_dir = args.model_dir
    model_fps = [model_fp for model_fp in model_dir.iterdir() 
                 if re.search(r'^model_\d.tar.gz', model_fp.name)]
    for model_fp in model_fps:
        model_archive = load_archive(str(model_fp), cuda_device=cuda)
        loaded_config = model_archive.config
        loaded_model = model_archive.model
        if 'dataset_reader' not in loaded_config:
            reader = DatasetReader.from_params(loaded_config['task_sentiment']['dataset_reader'])
        else:
            reader = DatasetReader.from_params(loaded_config['dataset_reader'])
        loaded_model.eval()

        predicted_tags: List[List[str]] = []
        for pred_tokens in create_input_sentences(evaluation_fp):
            instance = reader.text_to_instance(pred_tokens)
            pred_output = loaded_model.forward_on_instance(instance)
            tags = pred_output['tags']
            assert_err = (f'Number of predicted tags {len(pred_tokens)} should '
                          f'match the number of tokens being predicted {len(tags)}')
            assert len(tags) == len(pred_tokens), assert_err
            predicted_tags.append(tags)
        write_tags_to_file(save_fp, predicted_tags)
        print('done')