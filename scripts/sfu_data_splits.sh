#!/bin/bash
split_script="$(pwd)/multitask_negation_target/sfu_dataset_splitter.py"
sfu_split_dir="$(pwd)/data/auxiliary_tasks/en"
sfu_data="$(pwd)/data/auxiliary_tasks/en/SFU.conll"
python $split_script 0.8 0.1 0.1 $sfu_split_dir $sfu_data --set_random_seed 