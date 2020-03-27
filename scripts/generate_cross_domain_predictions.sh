#!/bin/bash
python_path="$(which python)"
echo "Python Path being used:"
echo $python_path
declare -a model_arr=("stl" "mtl/conan_doyle" "mtl/sfu" "mtl/sfu_spec" "mtl/u_pos" "mtl/dr" "mtl/lextag")
declare -a dataset_arr=("laptop" "restaurant" "MAMS" "mpqa")
for model in "${model_arr[@]}"
do
    for dataset in "${dataset_arr[@]}"
    do
        for alt_dataset in "${dataset_arr[@]}"
        do
            if [ $dataset != $alt_dataset ]
            then
                python ./scripts/generate.py "./data/models/en/"$model"/"$dataset"_contextualized" "./data/main_task/en/"$alt_dataset"/dev.conll" "./data/results/en/cross_domain/"$model"/"$dataset"/"$alt_dataset --cuda
                python ./scripts/generate.py "./data/models/en/"$model"/"$dataset"_contextualized" "./data/main_task/en/"$alt_dataset"/test.conll" "./data/results/en/cross_domain/"$model"/"$dataset"/"$alt_dataset --cuda
            fi
        done
    done
done