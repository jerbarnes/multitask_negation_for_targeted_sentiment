#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
## Laptop
### STL
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/stl/laptop_contextualized --cuda
### MTL SFU
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/sfu/laptop_contextualized --cuda
### MTL CD
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/conan_doyle/laptop_contextualized --cuda
### MTL SFU Spec
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu_spec/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/sfu_spec/laptop_contextualized --cuda
### MTL UPOS
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/u_pos/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/u_pos/laptop_contextualized --cuda
### MTL DR
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/dr/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/dr/laptop_contextualized --cuda
### MTL LEXTAG
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop_contextualized ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/lextag/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop_contextualized ./data/main_task/en/laptop/test.conll ./data/results/en/mtl/lextag/laptop_contextualized --cuda
## Restaurant
### STL
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/stl/restaurant_contextualized --cuda
### MTL SFU
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/sfu/restaurant_contextualized --cuda
### MTL CD
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/conan_doyle/restaurant_contextualized --cuda
### MTL SFU SPEC
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu_spec/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/sfu_spec/restaurant_contextualized --cuda
### MTL UPOS
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/u_pos/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/u_pos/restaurant_contextualized --cuda
### MTL DR
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/dr/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/dr/restaurant_contextualized --cuda
### MTL LEXTAG
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant_contextualized ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/lextag/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant_contextualized ./data/main_task/en/restaurant/test.conll ./data/results/en/mtl/lextag/restaurant_contextualized --cuda
## MAMS
### STL
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/stl/MAMS_contextualized --cuda
### MTL SFU
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/sfu/MAMS_contextualized --cuda
### MTL CD
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/conan_doyle/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/conan_doyle/MAMS_contextualized --cuda
### MTL SFU SPEC
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu_spec/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/sfu_spec/MAMS_contextualized --cuda
### MTL UPOS
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/u_pos/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/u_pos/MAMS_contextualized --cuda
### MTL DR
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/dr/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/dr/MAMS_contextualized --cuda
### MTL LEXTAG
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS_contextualized ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/lextag/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS_contextualized ./data/main_task/en/MAMS/test.conll ./data/results/en/mtl/lextag/MAMS_contextualized --cuda
## mpqa
### STL
python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/stl/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/stl/mpqa_contextualized --cuda
### MTL SFU
python ./scripts/generate.py ./data/models/en/mtl/sfu/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/sfu/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/sfu/mpqa_contextualized --cuda
### MTL CD
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/conan_doyle/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/conan_doyle/mpqa_contextualized --cuda
### MTL SFU SPEC
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/sfu_spec/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/sfu_spec/mpqa_contextualized --cuda
### MTL UPOS
python ./scripts/generate.py ./data/models/en/mtl/u_pos/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/u_pos/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/u_pos/mpqa_contextualized --cuda
### MTL DR
python ./scripts/generate.py ./data/models/en/mtl/dr/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/dr/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/dr/mpqa_contextualized --cuda
### MTL LEXTAG
python ./scripts/generate.py ./data/models/en/mtl/lextag/mpqa_contextualized ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/lextag/mpqa_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/mpqa_contextualized ./data/main_task/en/mpqa/test.conll ./data/results/en/mtl/lextag/mpqa_contextualized --cuda