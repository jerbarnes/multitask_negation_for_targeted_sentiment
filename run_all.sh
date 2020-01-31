#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# Baseline Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop 5
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant 5
# Conan Doyle Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop 5
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant 5
# SFU Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop 5
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant 5