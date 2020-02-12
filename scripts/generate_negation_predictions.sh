#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
python ./scripts/generate.py ./data/models/en/stl/laptop ./data/main_task/en/laptop/dev_neg.conll ./data/results/en/stl/laptop --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant ./data/main_task/en/restaurant/dev_neg.conll ./data/results/en/stl/restaurant --cuda

python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop ./data/main_task/en/laptop/dev_neg.conll ./data/results/en/mtl/conan_doyle/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant ./data/main_task/en/restaurant/dev_neg.conll ./data/results/en/mtl/conan_doyle/restaurant --cuda

python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop ./data/main_task/en/laptop/dev_neg.conll ./data/results/en/mtl/sfu/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant ./data/main_task/en/restaurant/dev_neg.conll ./data/results/en/mtl/sfu/restaurant --cuda