#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# Baseline Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop 5 ./data/models/en/stl/laptop
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant 5 ./data/models/en/stl/restaurant
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS 5 ./data/models/en/stl/MAMS
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/stl/mpqa 5 ./data/models/en/stl/mpqa
# Conan Doyle Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop 5 ./data/models/en/mtl/conan_doyle/laptop --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant 5 ./data/models/en/mtl/conan_doyle/restaurant --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/conan_doyle/MAMS 5 ./data/models/en/mtl/conan_doyle/MAMS --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/conan_doyle/mpqa 5 ./data/models/en/mtl/conan_doyle/mpqa --mtl
# SFU Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop 5 ./data/models/en/mtl/sfu/laptop --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant 5 ./data/models/en/mtl/sfu/restaurant --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu/MAMS 5 ./data/models/en/mtl/sfu/MAMS --mtl
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/sfu/mpqa 5 ./data/models/en/mtl/sfu/mpqa --mtl
# SFU Speculation
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu_spec/laptop 5 ./data/models/en/mtl/sfu_spec/laptop --mtl --aux_name speculation
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu_spec/restaurant 5 ./data/models/en/mtl/sfu_spec/restaurant --mtl --aux_name speculation
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu_spec/MAMS 5 ./data/models/en/mtl/sfu_spec/MAMS --mtl --aux_name speculation
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/sfu_spec/mpqa 5 ./data/models/en/mtl/sfu_spec/mpqa --mtl --aux_name speculation
# UPOS Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/u_pos/laptop 5 ./data/models/en/mtl/u_pos/laptop --mtl --aux_name upos
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/u_pos/restaurant 5 ./data/models/en/mtl/u_pos/restaurant --mtl --aux_name upos
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/u_pos/MAMS 5 ./data/models/en/mtl/u_pos/MAMS --mtl --aux_name upos
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/u_pos/mpqa 5 ./data/models/en/mtl/u_pos/mpqa --mtl --aux_name upos
# Dependency Relations Experiments
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/dr/laptop 5 ./data/models/en/mtl/dr/laptop --mtl --aux_name dr
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/dr/restaurant 5 ./data/models/en/mtl/dr/restaurant --mtl --aux_name dr
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/dr/MAMS 5 ./data/models/en/mtl/dr/MAMS --mtl --aux_name dr
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/dr/mpqa 5 ./data/models/en/mtl/dr/mpqa --mtl --aux_name dr
# LEXTAG
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/lextag/laptop 5 ./data/models/en/mtl/lextag/laptop --mtl --aux_name lextag
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/lextag/restaurant 5 ./data/models/en/mtl/lextag/restaurant --mtl --aux_name lextag
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/lextag/MAMS 5 ./data/models/en/mtl/lextag/MAMS --mtl --aux_name lextag
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/lextag/mpqa 5 ./data/models/en/mtl/lextag/mpqa --mtl --aux_name lextag
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
