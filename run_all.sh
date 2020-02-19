#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# Baseline Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop 5 ./data/models/en/stl/laptop
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant 5 ./data/models/en/stl/restaurant
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS 5 ./data/models/en/stl/MAMS
# Conan Doyle Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop 5 ./data/models/en/mtl/conan_doyle/laptop --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant 5 ./data/models/en/mtl/conan_doyle/restaurant --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/conan_doyle/MAMS 5 ./data/models/en/mtl/conan_doyle/MAMS --mtl
# SFU Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop 5 ./data/models/en/mtl/sfu/laptop --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant 5 ./data/models/en/mtl/sfu/restaurant --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu/MAMS 5 ./data/models/en/mtl/sfu/MAMS --mtl
# Baseline contextualized embeddings
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/restaurant_contextualized.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant_contextualized 5 ./data/models/en/stl/restaurant_contextualized
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop_contextualized.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop_contextualized 5 ./data/models/en/stl/laptop_contextualized
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mams_contextualized.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS_contextualized 5 ./data/models/en/stl/MAMS_contextualized
# Conan Doyle contextualized
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/laptop_contextualized.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop_contextualized 5 ./data/models/en/mtl/conan_doyle/laptop_contextualized --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/restaurant_contextualized.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant_contextualized 5 ./data/models/en/mtl/conan_doyle/restaurant_contextualized --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mams_contextualized.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/conan_doyle/MAMS_contextualized 5 ./data/models/en/mtl/conan_doyle/MAMS_contextualized --mtl
# SFU contextualized
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/laptop_contextualized.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop_contextualized 5 ./data/models/en/mtl/sfu/laptop_contextualized --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/restaurant_contextualized.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant_contextualized 5 ./data/models/en/mtl/sfu/restaurant_contextualized --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/mams_contextualized.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu/MAMS_contextualized 5 ./data/models/en/mtl/sfu/MAMS_contextualized --mtl
