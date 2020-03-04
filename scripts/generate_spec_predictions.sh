#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# STL
python ./scripts/generate.py ./data/models/en/stl/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/stl/laptop --cuda
# MTL CD
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/conan_doyle/laptop --cuda
# MTL SFU
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/sfu/laptop --cuda
# MTL SFU SPEC
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/sfu_spec/laptop --cuda
# MTL UPOS
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/u_pos/laptop --cuda
# MTL DR
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/dr/laptop --cuda
# MTL LEXTAG
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop ./data/main_task/en/laptop/dev_spec.conll ./data/results/en/mtl/lextag/laptop --cuda