import re
from pathlib import Path

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
for mtl_task_path, mtl_name in zip(mtl_tasks_paths, mtl_task_names):
    a_model = load_archive(str(mtl_task_path.resolve()))
    additional_params = count_crf_parameters(a_model.model)
    frac_additional_params = 100 * (float(additional_params) / stl_trainable_params)
    num_tags = a_model.model.num_tags
    # This is due to two tags that do not count which are padding and unknown 
    # which are auto added through the allennlp framework
    if mtl_name in ['UPOS', 'DR', 'LEX']:
        num_tags -= 2
    print(f'MTL task: {mtl_name}, the number of additional parameter the '
          f'mtl task adds to the overall model: {additional_params} '
          f'({frac_additional_params}%)\nNumber of tagging labels e.g. '
          f'BIO with labels {num_tags}')