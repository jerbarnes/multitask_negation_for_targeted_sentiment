# Multitask incorporation of negation for targeted sentiment classification


The idea is to use multitask learning to incorporate negation information into a targeted sentiment classifier

## Model
We could start with the BiLSTM + CRF model I used for sentence-level classification, but adapted to targeted sentiment. Other models could be cool too though.

The model currently is a BiLSTM + CRF model where BiLSTM has a 50 dimension hidden state.

## Resources
We assume any resource mentioned here is stored within `./resources`.
### Word Embeddings
All embeddings mentioned here have to be downloaded into their respective directories as they are too large for the repository.
#### English
Here we used the [300D 840B token GloVe embedding](https://nlp.stanford.edu/projects/glove/) of which we assume this can be found at the following path `./resources/embeddings/en/glove.840B.300d.txt`.

### Model Configurations
All the model configurations used in the main experiments can be found within `./resources/model_configs`.

## Datasets
# Sentiment
1. SemEval 2016 Aspect-based datasets (EN, ES)

The sentiment datasets need to be downloaded and converted from BIOSE to BIOUL format. The format change is only label differences e.g. S==U and L==E. The sentiment datasets are from [Li et al. 2019](https://www.aaai.org/ojs/index.php/AAAI/article/view/4643) and they are the SemEval 2014 Laptop dataset and the SemEval 2014, 2015, and 2016 combined Restaurant dataset. The downloaded datasets will go into the following folders `./data/main_task/en/laptop` and `./data/main_task/en/restaurant`. To download run the following script:
``` bash
python targeted_sentiment_downloader_converter.py
```

This script will also download the MAMS ATSA Restaurant dataset from [Jiang et al. 2019](https://www.aclweb.org/anthology/D19-1654.pdf) and convert it to CONLL format.

All the sentiment/main task data will be found in the `./data/main_task` folder.

# Negation
1. EN - [ConanDoyleNeg](https://www.aclweb.org/anthology/S12-1035.pdf), [SFU Review Corpus](https://www.aclweb.org/anthology/L12-1298/)
2. ES - SFU Spanish

For both the ConanDoyleNeg and SFU review corpus the negation scopes and cues are discontinous such that the BIO scheme does not continue if a cue is within a scope or vice versa, an example of this is shown below where the dis-continus version is example 1 (which is used in this project) and the continous version is shown in example 2.

#### Example 1
``` python
'''
Mooner	O	O
simply	O	O
forgot	O	O
to	O	O
show	O	O
up	O	O
for	O	O
his	O	O
court	O	O
appointment	O	O
,	O	O
as	O	O
if	O	B_speccue
he	O	B_spec
could	O	B_speccue
ever	O	B_spec
remember	O	I_spec
,	O	O
and	O	O
'''
```

#### Example 2
``` python
'''
Mooner	O	O
simply	O	O
forgot	O	O
to	O	O
show	O	O
up	O	O
for	O	O
his	O	O
court	O	O
appointment	O	O
,	O	O
as	O	O
if	O	B_speccue
he	O	I_spec
could	O	I_speccue
ever	O	I_spec
remember	O	I_spec
,	O	O
and	O	O
'''
```

## SFU Dataset Splitting
To create the 80%, 10%, and 10% train, development, and test splits for the [SFU dataset](./data/auxiliary_tasks/en/SFU.conll) run the following bash script which will create `SFU_train.conll`, `SFU_dev.conll`, and `SFU_test.conll` files within [`./data/auxiliary_tasks/en/`](./data/auxiliary_tasks/en/) directory:
``` bash
./scripts/sfu_data_splits.sh
```

## Negation and Speculation Dataset Statistics

The table below states the complete dataset statistics, if the dataset like Conan Doyle has specified train, development, and test splits these are all combined and the combined statistics are stated below.

| Dataset     | Label       | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|-------------|-------------|-------------------|---------------------|---------------|---------------------|
| SFU         | Negation    | 1,263 (2,156)     | 1,446 (8,215)       | 17,128        | 1,165               |
| SFU         | Speculation | 513 (562)         | 586 (3,483)         | 17,128        | 405                 |
| Conan Doyle | Negation    | 1,197 (1,222)     | 2,220 (9,761)       | 1,221         | 1,221               |

No. = Number

No. Label Sentences = Number of sentences from all of the sentences that contain at least one negation/speculation cue or scope token.

The number of scopes and cues states the number of complete BIO label spans where as the tokens defines the number of individual labels for instance [example 1](#example-1) from above contains 2 cues, 2 cue tokens, 2 scopes, and 3 scope tokens.

To generate the data statistics in the table above run the following bash script:
``` bash
./scripts/negation_statistics.sh
```

## Create dataset statistics for MPQA
``` bash
export STANFORDNLP_TEST_HOME=~/stanfordnlp_test
```

``` bash
allennlp dry-run ./resources/statistic_configs/en/mpqa.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

1094 Training sentences that contain at least one target out of the 4195 sentences.


## Create dataset statistics for the U-POS, X-POS, Dependency Relations, SMWE, and Super Sense tagging
In these auxilary tasks to get the vocabularly dataset statistics run the following:

For the Streusle we included empty nodes which are tokens that have a decimal numbered `ID`. The paper associated with the Streusle dataset is [A Corpus and Model Integrating Multiword Expressions and Supersenses by Schneider and Smith 2015](https://www.aclweb.org/anthology/N15-1177/).

``` bash
export STANFORDNLP_TEST_HOME=~/stanfordnlp_test
```

For U-POS
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_u_pos.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

For X-POS
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_x_pos.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

For Dependency Relations
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_dr.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

For SMWE
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_smwe.jsonnet -s /tmp/dry --include-package multitask_negation_target
```
Predicting True or False on the SMWE is highly in-balanced for the False class with 6987 in the True and 48598 in the False.

For SS
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_ss.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

For LEXTAG
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_lextag.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

### Dataset split statistics
The tables below states the dataset **split** statistics:
#### SFU Negation

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 1,018 (1,749)     | 1,155 (6,562)       | 13,712        | 934                 |
| Dev      | 121 (198)         | 154 (861)           | 1,713         | 114                 |
| Test     | 124 (209)         | 137 (792)           | 1,703         | 117                 |
| Combined | 1,263 (2,156)     | 1,446 (8,215)       | 17,128        | 1,165               |

To generate the data for this table above run `./scripts/negation_split_statistics.sh SFU negation`

#### SFU Speculation

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 390 (425)         | 446 (2,623)         | 13,712        | 309                 |
| Dev      | 58 (63)           | 66 (402)            | 1,713         | 45                  |
| Test     | 65 (74)           | 74 (458)            | 1,703         | 51                  |
| Combined | 513 (562)         | 586 (3,483)         | 17,128        | 405                 |

To generate the data for this table above run `./scripts/negation_split_statistics.sh SFU speculation`

#### Conan Doyle Negation

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 821 (838)         | 1,507 (6,756)       | 842           | 842                 |
| Dev      | 143 (146)         | 284 (1,283)         | 144           | 144                 |
| Test     | 233 (238)         | 429 (1,722)         | 235           | 235                 |
| Combined | 1,197 (1,222)     | 2,220 (9,761)       | 1,221         | 1,221               |

To generate the data for this table above run `./scripts/negation_split_statistics.sh conandoyle negation`

## Experiments
The single task model uses a 2 layer Bi-LSTM with a projection layer that goes into CRF decoding layer, further there is a skip connection between the embedding and the 2nd layer Bi-LSTM layer. The multi task model uses the same 2 layer Bi-LSTM model but the auxiliary tasks only have access to the embedding and first Bi-LSTM layer which then feeds into a task specific projection layer which then uses either Softmax or CRF for decoding.

Before running any of the experiments for the single and multi task models we perform a hyperparameter search for both models.

### Hyperparameter tuning
The tuning is performed on the smallest datasets which are the Laptop dataset for the Sentiment/Main task and the Conan Doyle for the Negation/Auxiliary task when tuning the multi and single task models. The parameters we tune for are the following:
1. Dropout rate - between 0 and 0.5
2. Hidden size for shared/first layer of the Bi-LSTM  - between 30 and 110
3. Starting learning rate for adam - between 0.01 (1e-2) and 0.0001 (1e-4)

The tuning is performed separately for the single and multi-task models. The single task model will only be tuned for the sentiment task and not the negation. Furthermore we tune the models by randomly sampling the parameters stated above within the range specified changining the random seed each time, of which these parameters are sampled 30 times in total for each model. From the 30 model runs the parameters from the best run based on the F1-Span measure from the validation set are selected for all of the experiments for that model.

#### Multi Task Learning Tuning

As [stanford-nlp](https://github.com/stanfordnlp/stanfordnlp) package is installed and due to [Ray](https://ray.readthedocs.io/en/latest/) that is used in [allentune](https://github.com/allenai/allentune), the `STANFORDNLP_TEST_HOME` environment variable has to be set before using `allentune` thus I did the following:
``` bash
export STANFORDNLP_TEST_HOME=~/stanfordnlp_test
```

Run the following:
``` bash
allentune search \
    --experiment-name multi_task_laptop_conan_search \
    --num-cpus 5 \
    --num-gpus 1 \
    --cpus-per-trial 5 \
    --gpus-per-trial 1 \
    --search-space resources/tuning/tuning_configs/multi_task_search_space.json \
    --num-samples 30 \
    --base-config resources/tuning/tuning_configs/multi_task_laptop_conan.jsonnet \
    --include-package multitask_negation_target
allentune report \
    --log-dir logs/multi_task_laptop_conan_search/ \
    --performance-metric best_validation_f1-measure-overall \
    --model multi-task
allentune plot \
    --data-name Laptop \
    --subplot 1 1 \
    --figsize 10 10 \
    --result-file logs/multi_task_laptop_conan_search/results.jsonl \
    --output-file resources/tuning/multi_task_tuning_laptop_performance.pdf \
    --performance-metric-field best_validation_f1-measure-overall \
    --performance-metric F1-Span
```
The multi-task model found the following as the best parameters from run number 24 with a validation F1-Span score of 60.17%:
1. lr = 0.0019
2. shared/first layer hidden size = 65
3. dropout = 0.27
Of which the plot of the F1-Span metric on the validation set against the number of runs can be seen [here](./resources/tuning/multi_task_tuning_laptop_performance.pdf).

#### Single Task Learning Tuning

Single Task Laptop
Run the following:
``` bash
allentune search \
    --experiment-name single_task_laptop_search \
    --num-cpus 5 \
    --num-gpus 1 \
    --cpus-per-trial 5 \
    --gpus-per-trial 1 \
    --search-space resources/tuning/tuning_configs/single_task_search_space.json \
    --num-samples 30 \
    --base-config resources/tuning/tuning_configs/single_task_laptop.jsonnet \
    --include-package multitask_negation_target
allentune report \
    --log-dir logs/single_task_laptop_search/ \
    --performance-metric best_validation_f1-measure-overall \
    --model single-task
allentune plot \
    --data-name Laptop \
    --subplot 1 1 \
    --figsize 10 10 \
    --result-file logs/single_task_laptop_search/results.jsonl \
    --output-file resources/tuning/single_task_tuning_laptop_performance.pdf \
    --performance-metric-field best_validation_f1-measure-overall \
    --performance-metric F1-Span
```

The single-task model found the following as the best parameters from run number 7 with a validation F1-Span score of 61.56%:
1. lr = 0.0015
2. shared/first layer hidden size = 60
3. dropout = 0.5
Of which the plot of the F1-Span metric on the validation set against the number of runs can be seen [here](./resources/tuning/single_task_tuning_laptop_performance.pdf).

### Example of how to run the Single-Task System
You can use the allennlp train command here:
```
allennlp train resources/model_configs/targeted_sentiment_laptop_baseline.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Example of how to run the Multi-Task System
You can use the allennlp train command here:
```
allennlp train resources/model_configs/multi_task_trainer.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Single task models
In all of the experiments the python script has the following argument signature:
1. Model config file path
2. Main task test data file path
3. Main task development/validation data file path
4. Folder to save the results too. This folder will contain two files a `test.conll` and `dev.conll` each of these files will contain the predicted results for the associated data split. The files will have the following structure: `Token#Gold Label#Predicted Label 1#Predicted Label 2`. Where the `#` indicates whitespace and the number of predicted labels is determined by the number of times the model has been ran.
5. Number of times to run the model to overcome the random seed problem. In all of the experiments below they are ran 5 times.

#### Targeted Sentiment

For the laptop dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop 5 ./data/models/en/stl/laptop
```
For the Restaurant dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/stl/restaurant 5 ./data/models/en/stl/restaurant
```

For the MAMS dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/stl/MAMS 5 ./data/models/en/stl/MAMS
```

For the MPQA dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/stl/mpqa 5 ./data/models/en/stl/mpqa
```

To run that model:
``` bash
allennlp train resources/model_configs/stl/en/mpqa.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Multi task models
#### Conan Doyle
For the laptop dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/conan_doyle/laptop 5 ./data/models/en/mtl/conan_doyle/laptop --mtl
```
For the Restaurant dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/conan_doyle/restaurant 5 ./data/models/en/mtl/conan_doyle/restaurant --mtl
```

For the MAMS dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/conan_doyle/MAMS 5 ./data/models/en/mtl/conan_doyle/MAMS --mtl
```

For the MPQA dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/conan_doyle/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/conan_doyle/mpqa 5 ./data/models/en/mtl/conan_doyle/mpqa --mtl
```

#### SFU (Negation)
For the laptop dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu/laptop 5 ./data/models/en/mtl/sfu/laptop --mtl
```
For the Restaurant dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu/restaurant 5 ./data/models/en/mtl/sfu/restaurant --mtl
```
For the MAMS dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu/MAMS 5 ./data/models/en/mtl/sfu/MAMS --mtl
```

#### POS tagging (Streusle data)
Here the task is Universal POS tagging. When running the model once with one Bi-LSTM layer with a CRF decoder the POS tagging accuracy for test and validation respectively is:

`94.3% and 94.26%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/u_pos/pos.jsonnet -s /tmp/any --include-package multitask_negation_target
```

For the multi task learning models run the following for the respective datasets:

Laptop
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/u_pos/laptop 5 ./data/models/en/mtl/u_pos/laptop --mtl --aux_name upos
```

Restaurant
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/u_pos/restaurant 5 ./data/models/en/mtl/u_pos/restaurant --mtl --aux_name upos
```

MAMS
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/u_pos/MAMS 5 ./data/models/en/mtl/u_pos/MAMS --mtl --aux_name upos
```

For the MPQA dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/u_pos/mpqa 5 ./data/models/en/mtl/u_pos/mpqa --mtl --aux_name upos
```

#### Dependency Relation tagging (Streusle data)
Here the task is Dependency Relation tagging where we want to predict the dependency relation tag for a given token but not the dependency graph. When running the model once with one Bi-LSTM layer with a CRF decoder the Dependency Relation tagging accuracy for test and validation respectively is:

`88.22% and 87.49%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/dr/dr.jsonnet -s /tmp/any --include-package multitask_negation_target
```

For the multi task learning models run the following for the respective datasets:

Laptop
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/dr/laptop 5 ./data/models/en/mtl/dr/laptop --mtl --aux_name dr
```

Restaurant
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/dr/restaurant 5 ./data/models/en/mtl/dr/restaurant --mtl --aux_name dr
```

MAMS
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/dr/MAMS 5 ./data/models/en/mtl/dr/MAMS --mtl --aux_name dr
```

MPQA
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/dr/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/dr/mpqa 5 ./data/models/en/mtl/dr/mpqa --mtl --aux_name dr
```

#### Lexical tagging (Streusle data)
This is a complex BIO tagging task. Accuracy for test and validation respectively is:

`76.94% and 77.94%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/lextag/lextag.jsonnet -s /tmp/any --include-package multitask_negation_target
```

For the multi task learning models run the following for the respective datasets:

Laptop
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/lextag/laptop 5 ./data/models/en/mtl/lextag/laptop --mtl --aux_name lextag
```

Restaurant
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/lextag/restaurant 5 ./data/models/en/mtl/lextag/restaurant --mtl --aux_name lextag
```

MAMS
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/lextag/MAMS 5 ./data/models/en/mtl/lextag/MAMS --mtl --aux_name lextag
```

mpqa
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/lextag/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/lextag/mpqa 5 ./data/models/en/mtl/lextag/mpqa --mtl --aux_name lextag
```

#### SFU (Speculation)
Baseline F1 Spec scores for test and validation:

`42.9 and 56.6`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/sfu_spec/spec.jsonnet -s /tmp/any --include-package multitask_negation_target
```

For the laptop dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/mtl/sfu_spec/laptop 5 ./data/models/en/mtl/sfu_spec/laptop --mtl --aux_name speculation
```
For the Restaurant dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/restaurant.jsonnet ./data/main_task/en/restaurant/test.conll ./data/main_task/en/restaurant/dev.conll ./data/results/en/mtl/sfu_spec/restaurant 5 ./data/models/en/mtl/sfu_spec/restaurant --mtl --aux_name speculation
```
For the MAMS dataset:
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/mams.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/sfu_spec/MAMS 5 ./data/models/en/mtl/sfu_spec/MAMS --mtl --aux_name speculation
```
For the MPQA dataset
```
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/sfu_spec/mpqa.jsonnet ./data/main_task/en/mpqa/test.conll ./data/main_task/en/mpqa/dev.conll ./data/results/en/mtl/sfu_spec/mpqa 5 ./data/models/en/mtl/sfu_spec/mpqa --mtl --aux_name speculation
```

To run all of the experiments use the following script:
```
./run_all.sh
```

### Predicting on the Negation corpus
The Laptop and Restaurant development/validation and test dataset splits have been re-annotated so that targets have been negated when possible, thus there are still samples in these splits that do not have any form of negation. These splits can therefore to some extent test how well the models perform on a large amount of negated target data. Therefore both the MTL and STL models that were trained in the previous section will now be tested on these two splits. To get these result run the following:
``` bash
./scripts/generate_negation_predictions.sh
```

The negation corpus has also been filtered so that samples have to contain negation. This is to better isolate the negation phenonma, and therefore to better test the models capability on modelling negated samples. The script below runs the models on this smaller but more isolated negated data:
``` bash
./scripts/generate_negation_only_predictions.sh
```

The non-filtered negation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_neg.conll) and [Test](./data/main_task/en/laptop/test_neg.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_neg.conll) and [Test](./data/main_task/en/restaurant/test_neg.conll)

The filtered negation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_neg_only.conll) and [Test](./data/main_task/en/laptop/test_neg_only.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_neg_only.conll) and [Test](./data/main_task/en/restaurant/test_neg_only.conll)

### Predicting on the Speculation Corpus
The Laptop and Restaurant development/validation dataset split have been re-annotated so that targets have speculation added when possible, thus there are still samples in these splits that do not have any form of speculation. These splits can therefore to some extent test how well the models perform on a large amount of speculated target data. Therefore both the MTL and STL models that were trained in the previous section will now be tested on these two splits. To get these result run the following:

``` bash
./scripts/generate_spec_predictions.sh
```

The speculation corpus has also been filtered so that samples have to contain speculation. This is to better isolate the speculation phenonma, and therefore to better test the models capability on modelling speculative samples. The script below runs the models on this smaller but more isolated speculative data:
``` bash
./scripts/generate_spec_only_predictions.sh
```

The non-filtered speculation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_spec.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_spec.conll)

The filtered speculation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_spec_only.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_spec_only.conll)

### Predicting on cross domain
For each of the CWR we use them to predict on the other domains that they were not trained on.
```
./scripts/generate_cross_domain_predictions.sh
```

As the MPQA trained models and dataset use a different sentiment label mapping (they use positive, neutral, and negative labels instead of POS, NEU, and NEG respectively that all other models and datasets use) we need to change the labels so they are all the same. To do this run the following command once and only once:
``` bash
python scripts/cross_domain_change_labels.py True
```

As the generation of the cross domain results takes a long time on the Google Colab notebook, we generate a JSON file of the results that can be used in the Google Colab notebook, to create the results:
``` bash
python scripts/cross_domain_change_labels.py False
```
The JSON file is saved at `data/results/en/cross_domain/results.json`.

The Google Colab notebook of the analysis for this can be seen [here.](./notebooks/Cross_Domain.ipynb)

### Creating detailed sentiment analysis results
The results in the paper that are associated with Distinct Sentiment (DS) and Strict Text ACcuracy (STAC) are created using the following script. The results that the script generates for those metrics/subsets are stored in the following JSON file.

``` bash
python scripts/generate_detailed_sentiment_results.py ./data/results/en/ ./data/results/detailed_sentiment_results.json
```

The results are then displayed in the following [notebook.](./notebooks/Multi_task_detailed_sentiment_metrics.ipynb)

### Number of parameters
To find the number of additional parameters the MTL models add, run the following:
``` bash
python number_parameters.py
```

## Requirements

1. Python >= 3.6
2. `pip install .` 
3. If you are developing or want to run the tests `pip install -r requirements.txt`
3. sklearn  ```pip install -U scikit-learn```
4. Pytorch ```pip install torch torchvision```

## Run the tests:
`python -m pytest`

## Other ideas
1. Having a shared and task specific Bi-LSTM layers at the moment all Bi-LSTM layers are shared
2. The affect of having very little sentiment data and all of the negated data
3. The affect of pre-trained word embeddings, if random does multi-task/transfer help a lot more.
