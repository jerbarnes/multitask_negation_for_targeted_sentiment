#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# STL
## Laptop
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/stl/laptop_contextualized --cuda
## Restaurant
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/stl/restaurant_contextualized --cuda
## MAMS
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/stl/MAMS_contextualized --cuda
## mpqa
python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/stl/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/stl/mpqa_contextualized --cuda