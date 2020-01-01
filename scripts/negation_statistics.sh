#!/bin/bash
data_dir="$(pwd)/data/auxiliary_tasks/en"
sfu_data="$data_dir/SFU.conll"
analysis_script="$(pwd)/multitask_negation_target/analysis/negation_dataset_analysis.py"
python $analysis_script $sfu_data "sfu" "negation"
python $analysis_script $sfu_data "sfu" "speculation"
conandoyle_data="$data_dir/conandoyle_all.conllu"
python $analysis_script $conandoyle_data "conandoyle" "negation"