from pathlib import Path
from typing import List, Dict
import tempfile
import timeit

import allennlp
from allennlp.models.archival import load_archive
from target_extraction.allen.allennlp_model import AllenNLPModel
import pandas as pd

import multitask_negation_target

def run_model(model, data, batch_size) -> List:
    stored_preds = []
    for value in model._predict_iter(data, batch_size=batch_size):
        stored_preds.append(value['tags'])
    return stored_preds

def model_performance(model_path: Path, batch_size: int, 
                      data: List[Dict[str, str]], cuda_device: int) -> List[float]:
    loaded_model = load_archive(str(model_path.resolve()), cuda_device=cuda_device)
    loaded_model_config = loaded_model.config
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fp = Path(temp_dir, 'temp_file')
        loaded_model_config.to_file(str(temp_fp.resolve()))
        
        allen_model = AllenNLPModel(name='example', model_param_fp=temp_fp, 
                                    predictor_name='sentence-tagger')
        allen_model.model = loaded_model.model
        times = timeit.repeat("run_model(allen_model, data, batch_size)", globals=locals(),
                              setup="from __main__ import run_model", number=1, repeat=3)
        return times
        

laptop_test_data = Path('./data/main_task/en/laptop/test.conll')
laptop_sentences: List[Dict[str, str]] = []
with laptop_test_data.open('r') as laptop_test_file:
    tokens = []
    for line in laptop_test_file:
        if not line.strip():
            if not tokens:
                raise ValueError('These has to be tokens to create a sentence')
            laptop_sentences.append({'sentence': ' '.join(tokens)})
            tokens = []
        else:
            tokens.append(line.strip().split()[0])
    if tokens:
        laptop_sentences.append({'sentence': ' '.join(tokens)})
# Create a table of Embedding, Model, Batch Size, Lowest time, Highest time

stl_cwr_model_path = Path('./data/models/en/stl/laptop_contextualized/model_0.tar.gz')
stl_model_path = Path('./data/models/en/stl/laptop/model_0.tar.gz')
mtl_cwr_model_path = Path('./data/models/en/mtl/sfu/laptop_contextualized/model_0.tar.gz')
mtl_model_path = Path('./data/models/en/mtl/sfu/laptop/model_0.tar.gz')

embeddings = []
model_names = []
batch_sizes = []
devices = []
min_times = []
max_times = []
device_mapper = {-1: 'CPU', 0: 'GPU'}

model_dict = {'CWR': {'STL': stl_cwr_model_path, 'MTL': mtl_cwr_model_path},
              'GloVe': {'STL': stl_model_path, 'MTL': mtl_model_path}}
for embedding_name, model_data in model_dict.items():
    for model_name, model_path in model_data.items():
        for batch_size in [1, 8, 16, 32]:
            for device in [-1, 0]:
                # GPU = 0 and CPU = -1
                times = model_performance(model_path, batch_size, 
                                          laptop_sentences, device)
                min_times.append(min(times))
                max_times.append(max(times))
                devices.append(device_mapper[device])
                batch_sizes.append(batch_size)
                model_names.append(model_name)
                embeddings.append(embedding_name)

inference_df = {'Embedding': embeddings, 'Model': model_names, 
                'Batch Size': batch_sizes, 'Device': devices, 
                'Min Time': min_times, 'Max Time': max_times}
with Path('inference_save.json').open('w+') as fp:
    import json
    json.dump(inference_df, fp)
import pandas as pd
df = pd.DataFrame(inference_df)
print(df.to_latex())