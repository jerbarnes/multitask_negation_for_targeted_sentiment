#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# STL Only
## Laptop, using laptop models to predict on Restaurant, MAMS, and MPQA
### Restaurant
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/cross_domain/stl/laptop/restaurant --cuda
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/cross_domain/stl/laptop/restaurant --cuda
### MAMS
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/cross_domain/stl/laptop/MAMS --cuda
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/cross_domain/stl/laptop/MAMS --cuda
### mpqa
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/cross_domain/stl/laptop/mpqa --cuda
#python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/cross_domain/stl/laptop/mpqa --cuda

## Restaurant, using restaurant models to predict on Laptop, MAMS, and MPQA
### Laptop
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/cross_domain/stl/restaurant/laptop --cuda
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/cross_domain/stl/restaurant/laptop --cuda
### MAMS
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/cross_domain/stl/restaurant/MAMS --cuda
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/cross_domain/stl/restaurant/MAMS --cuda
### mpqa
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/cross_domain/stl/restaurant/mpqa --cuda
#python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/cross_domain/stl/restaurant/mpqa --cuda

## MAMS, using MAMS models to predict on Laptop, Restaurant, and MPQA
### Restaurant
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/cross_domain/stl/MAMS/restaurant --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/cross_domain/stl/MAMS/restaurant --cuda
### Laptop
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/cross_domain/stl/MAMS/laptop --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/cross_domain/stl/MAMS/laptop --cuda
### mpqa
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/cross_domain/stl/MAMS/mpqa --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/cross_domain/stl/MAMS/mpqa --cuda

## mpqa, using mpqa models to predict on Laptop, MAMS, and Restaurant
### Restaurant
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/cross_domain/stl/mpqa/restaurant --cuda
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/cross_domain/stl/mpqa/restaurant --cuda
### MAMS
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/cross_domain/stl/mpqa/MAMS --cuda
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/cross_domain/stl/mpqa/MAMS --cuda
### Laptop
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/cross_domain/stl/mpqa/laptop --cuda
#python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/cross_domain/stl/mpqa/laptop --cuda