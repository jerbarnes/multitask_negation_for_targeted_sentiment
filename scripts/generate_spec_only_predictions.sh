#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
# STL
## GLOVE
python ./scripts/generate.py ./data/models/en/stl/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/stl/laptop --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/stl/restaurant --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/stl/MAMS --cuda
python ./scripts/generate.py ./data/models/en/stl/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/stl/laptop --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/stl/restaurant --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/stl/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/stl/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/stl/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/stl/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/stl/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/stl/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/stl/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/stl/MAMS_contextualized --cuda
# MTL CD
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/conan_doyle/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/conan_doyle/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/conan_doyle/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/conan_doyle/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/conan_doyle/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/conan_doyle/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/conan_doyle/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/conan_doyle/MAMS_contextualized --cuda
# MTL SFU
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/sfu/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/sfu/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/sfu/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/sfu/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu/MAMS_contextualized --cuda
# MTL SFU SPEC
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/sfu_spec/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu_spec/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu_spec/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/sfu_spec/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/sfu_spec/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu_spec/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/sfu_spec/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/sfu_spec/MAMS_contextualized --cuda
# MTL UPOS
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/u_pos/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/u_pos/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/u_pos/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/u_pos/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/u_pos/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/u_pos/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/u_pos/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/u_pos/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/u_pos/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/u_pos/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/u_pos/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/u_pos/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/u_pos/MAMS_contextualized --cuda
# MTL DR
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/dr/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/dr/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/dr/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/dr/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/dr/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/dr/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/dr/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/dr/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/dr/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/dr/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/dr/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/dr/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/dr/MAMS_contextualized --cuda
# MTL LEXTAG
## GLOVE
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/lextag/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/lextag/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/lextag/MAMS --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/lextag/laptop --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/lextag/restaurant --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/lextag/MAMS --cuda
## CWR
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop_contextualized ./data/main_task/en/laptop/dev_spec_only.conll ./data/results/en/mtl/lextag/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/lextag/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS_contextualized ./data/main_task/en/restaurant/dev_spec_only.conll ./data/results/en/mtl/lextag/MAMS_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/laptop_contextualized ./data/main_task/en/laptop/test_spec_only.conll ./data/results/en/mtl/lextag/laptop_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/restaurant_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/lextag/restaurant_contextualized --cuda
python ./scripts/generate.py ./data/models/en/mtl/lextag/MAMS_contextualized ./data/main_task/en/restaurant/test_spec_only.conll ./data/results/en/mtl/lextag/MAMS_contextualized --cuda