import re
from pathlib import Path
from typing import Dict

from allennlp.models.archival import load_archive

import multitask_negation_target

def count_crf_parameters(model):
    '''
    Reference:
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    '''
    crf_params = sum(p.numel() for name, p in model.named_parameters() if re.search(r'^crf.+', name))
    tag_params = sum(p.numel() for name, p in model.named_parameters() if re.search(r'^tag_projection_lay.+', name))
    return crf_params + tag_params

def count_parameters(model, trainable: bool = False):
    '''
    Reference:
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    '''
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

stl_model = Path("./data/models/en/stl/laptop/model_0.tar.gz")
a_model = load_archive(str(stl_model.resolve()))
stl_params = count_parameters(a_model.model)
stl_trainable_params = count_parameters(a_model.model, True)
print(f'Number of parameter in the STL model {stl_params}, '
      f'trainable params {stl_trainable_params}')

mtl_dir = Path("./data/models/en/mtl").resolve()
mtl_tasks_paths = [mtl_dir / "conan_doyle/laptop/task_negation_model_0.tar.gz", 
                   mtl_dir / "sfu/laptop/task_negation_model_0.tar.gz",
                   mtl_dir / "sfu_spec/laptop/task_speculation_model_0.tar.gz",
                   mtl_dir / "u_pos/laptop/task_upos_model_0.tar.gz",
                   mtl_dir / "dr/laptop/task_dr_model_0.tar.gz",
                   mtl_dir / "lextag/laptop/task_lextag_model_0.tar.gz"]
mtl_task_names = ['CD', 'SFU', 'SPEC', 'UPOS', 'DR', 'LEX']
mtl_task_additional_params: Dict[str, int] = {}
for mtl_task_path, mtl_name in zip(mtl_tasks_paths, mtl_task_names):
    a_model = load_archive(str(mtl_task_path.resolve()))
    additional_params = count_crf_parameters(a_model.model)
    frac_additional_params = 100 * (float(additional_params) / stl_trainable_params)
    num_tags = a_model.model.num_tags
    # This is due to two tags that do not count which are padding and unknown 
    # which are auto added through the allennlp framework
    if mtl_name in ['UPOS', 'DR', 'LEX']:
        num_tags -= 2
    mtl_task_additional_params[mtl_name] = additional_params
    print(f'MTL task: {mtl_name}, the number of additional parameter the '
          f'mtl task adds to the overall model: {additional_params} '
          f'({frac_additional_params}%)\nNumber of tagging labels e.g. '
          f'BIO with labels {num_tags}')
# This is used to create the parameters table.
print('\nParameter table data for both MTL parameters with auxiliary task '
      'parameters and without auxiliary task parameters:')
mtl_tasks_paths = [mtl_dir / "conan_doyle/laptop/model_0.tar.gz", 
                      mtl_dir / "sfu/laptop/model_0.tar.gz",
                      mtl_dir / "sfu_spec/laptop/model_0.tar.gz",
                      mtl_dir / "u_pos/laptop/model_0.tar.gz",
                      mtl_dir / "dr/laptop/model_0.tar.gz",
                      mtl_dir / "lextag/laptop/model_0.tar.gz"]
mtl_cwr_task_paths = [mtl_dir / "conan_doyle/laptop_contextualized/model_0.tar.gz", 
                      mtl_dir / "sfu/laptop_contextualized/model_0.tar.gz",
                      mtl_dir / "sfu_spec/laptop_contextualized/model_0.tar.gz",
                      mtl_dir / "u_pos/laptop_contextualized/model_0.tar.gz",
                      mtl_dir / "dr/laptop_contextualized/model_0.tar.gz",
                      mtl_dir / "lextag/laptop_contextualized/model_0.tar.gz"]

def param_string(num_train_params: int, num_all_params: int, 
                 embedding_name: str, model_name: str) -> str:
    param_str = (f'Embedding: {embedding_name}, model: {model_name}, '
                 f'Total number parameters: {num_all_params}, '
                 f'Total Trainable Parameters: {num_train_params}')
    return param_str
stl_cwr_model = Path("./data/models/en/stl/laptop_contextualized/model_0.tar.gz")
mtl_embedding_paths = {'GloVe': mtl_tasks_paths, 'CWR': mtl_cwr_task_paths}
stl_embedding_paths = {'GloVe': stl_model, 'CWR': stl_cwr_model}
for embedding_name, mtl_paths in mtl_embedding_paths.items():
    for mtl_name, mtl_path in zip(mtl_task_names, mtl_paths):
        a_model = load_archive(str(mtl_path.resolve()))
        all_params = count_parameters(a_model.model, trainable=False)
        trainable_params = count_parameters(a_model.model, trainable=True)
        print('Without auxiliary task parameters:')
        print(param_string(trainable_params, all_params, embedding_name, mtl_name))
        print('WITH auxiliary task parameters:')
        trainable_params += mtl_task_additional_params[mtl_name]
        all_params += mtl_task_additional_params[mtl_name]
        print(param_string(trainable_params, all_params, embedding_name, mtl_name))
        print('\n')
    a_model = load_archive(str(stl_embedding_paths[embedding_name].resolve()))
    all_params = count_parameters(a_model.model, trainable=False)
    trainable_params = count_parameters(a_model.model, trainable=True)
    print(param_string(trainable_params, all_params, embedding_name, 'STL'))
    print('\n')
#task_paths = ['conan_doyle', 'dr', 'lextag', 'sfu', 'sfu_spec', 'u_pos']
#dataset_names = ['laptop', 'MAMS', 'restaurant', 'mpqa']
#for task_name in task_paths:
#    for dataset_name in dataset_names:
#        base_mtl_model_path = Path(mtl_dir, task_name, f'{dataset_name}_contextualized')
#        for i in range(5):
#            mtl_model_path = Path(base_mtl_model_path, f'model_{i}.tar.gz')
#            a_model = load_archive(str(mtl_model_path.resolve()))
#            all_params = count_parameters(a_model.model, trainable=False)
#            trainable_params = count_parameters(a_model.model, trainable=True)
#            print(param_string(trainable_params, all_params, 'CWR', f'{task_name} {dataset_name} {i}'))
#a_model = load_archive(str(stl_cwr_model.resolve())).model
#param_name = [(name, param) for name, param in a_model.named_parameters()]
#embedding_params = [(name, param) for name, param in a_model.named_parameters() if re.search(r'^text_field_embedder.token_embedder_elmo.+', name)]
#embedding_params = [(name, param) for name, param in embedding_params if param.requires_grad]
#embedding_params_num = [param.numel() for name, param in embedding_params]
#print('here')