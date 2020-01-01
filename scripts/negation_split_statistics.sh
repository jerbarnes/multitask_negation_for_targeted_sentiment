#!/bin/bash
dataset_name=$1
label_type=$2

data_dir="$(pwd)/data/auxiliary_tasks/en"



train_data="$data_dir/"$dataset_name"_train.conll"
dev_data="$data_dir/"$dataset_name"_dev.conll"
test_data="$data_dir/"$dataset_name"_test.conll"

analysis_script="$(pwd)/multitask_negation_target/analysis/negation_dataset_analysis.py"

declare -a data_arr=("train" "dev" "test")

for data in "${data_arr[@]}"
do
    echo $data
    data_file="$data_dir/"$dataset_name"_"$data".conll"
    if [ $dataset_name = 'conandoyle' ]; then
        data_file="$data_dir/"$dataset_name"_"$data".conllu"
    fi
    lower_dataset_name="${dataset_name,,}"
    python $analysis_script $data_file $lower_dataset_name $label_type
done


