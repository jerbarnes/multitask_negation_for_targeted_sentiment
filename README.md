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

## The Negated and Speculative Aspect Based Sentiment Analysis datasets

The Development and Test splits for the negated and speculative only Aspect Based Sentiment Analysis datasets that have been annotated by one of the authors of this work can be found:

1. Laptop<sub>*Neg*</sub> -- [Development](./data/main_task/en/laptop/dev_neg_only.conll), [Test](./data/main_task/en/laptop/test_neg_only.conll)
2. Laptop<sub>*Spec*</sub> -- [Development](./data/main_task/en/laptop/dev_spec_only.conll), [Test](./data/main_task/en/laptop/test_spec_only.conll)
3. Restaurant<sub>*Neg*</sub> -- [Development](./data/main_task/en/restaurant/dev_neg_only.conll), [Test](./data/main_task/en/restaurant/test_neg_only.conll)
4. Restaurant<sub>*Spec*</sub> -- [Development](./data/main_task/en/restaurant/dev_spec_only.conll), [Test](./data/main_task/en/restaurant/test_spec_only.conll)

Within these datasets only negated or speculative sentiments exist.

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
```
The multi-task model found the following as the best parameters from run number 24 with a validation F1-Span score of 60.17%:
1. lr = 0.0019
2. shared/first layer hidden size = 65
3. dropout = 0.27

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
```

The single-task model found the following as the best parameters from run number 7 with a validation F1-Span score of 61.56%:
1. lr = 0.0015
2. shared/first layer hidden size = 60
3. dropout = 0.5

#### Plotting the expected validation score
To get a plot of the two STL and MTL models expected validation scores, you first have to copy the results from the [STL](./logs/single_task_laptop_search/results.jsonl) and [MTL](./logs/multi_task_laptop_conan_search/results.jsonl) together into a new file. Of which we have done this [here](./logs/other_result.jsonl). With this new combined file run the following to create the plot, which can be found [here](./resources/tuning/combined_tuning_laptop_performance.pdf):
``` bash
allentune plot \
    --data-name Laptop \
    --subplots 1 1 \
    --figsize 10 10 \
    --plot-errorbar \
    --result-file logs/other_result.jsonl \
    --output-file resources/tuning/combined_tuning_laptop_performance.pdf \
    --performance-metric-field best_validation_f1-measure-overall \
    --performance-metric F1-Span
```

### Example of how to Train the Single-Task System using AllenNLP train command
You can use the allennlp train command here:
```
allennlp train resources/model_configs/targeted_sentiment_laptop_baseline.jsonnet -s /tmp/any --include-package multitask_negation_target
```
### Example of how to Train the Multi-Task System using AllenNLP train command
You can use the allennlp train command here:
```
allennlp train resources/model_configs/multi_task_trainer.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Mass experiments setup
The previous two subsections describe how to just train one model on one dataset, in the paper we trained each model 5 times and there were numerous models (1 STL and 6 MTL) and 4 datasets. Thus to do this we created two scripts. The first script trains a model e.g. STL on one dataset 5 times and then saves the 5 models including the respective auxiliary task models where applicable and also saves the result. The second script runs the first script across all of the models and datasets.

The first python script has the following argument signature:
1. Model config file path
2. Main task test data file path
3. Main task development/validation data file path
4. Folder to save the results too. This folder will contain two files a `test.conll` and `dev.conll` each of these files will contain the predicted results for the associated data split. The files will have the following structure: `Token#Gold Label#Predicted Label 1#Predicted Label 2`. Where the `#` indicates whitespace and the number of predicted labels is determined by the number of times the model has been ran.
5. Number of times to run the model -- in all of our experiments we run the model 5 times thus this is always 5 in our case.
6. Folder to save the trained model(s) too. If you are training an MTL model then the auxiliary task model(s) will also be saved here.
7. OPTIONAL FLAG `--mtl` is required if you are training an MTL model.
8. OPTIONAL FLAG `--aux_name` the name of auxilary task is required if training an MTL model. By default this is `negation` but if a `negation` task is not being trained than the name of the task from the model config is required e.g. for u_pos the task name is `task_u_pos` thus you remove the `task_` to get the `aux_name` which in this case is `u_pos`.

And an example of running this script is shown below, whereby this runs the STL model with GloVe vectors 5 times on the Laptop dataset:

``` bash
python ./scripts/train_and_generate.py ./resources/model_configs/stl/en/laptop.jsonnet ./data/main_task/en/laptop/test.conll ./data/main_task/en/laptop/dev.conll ./data/results/en/stl/laptop 5 ./data/models/en/stl/laptop
```

The MTL models can be run in a similar way but does require a few extra flags. Thus the example below shows the MTL (UPOS) model run 5 times with CWR on the MAMS dataset:

``` bash
python ./scripts/train_and_generate.py ./resources/model_configs/mtl/en/u_pos/mams_contextualized.jsonnet ./data/main_task/en/MAMS/test.conll ./data/main_task/en/MAMS/dev.conll ./data/results/en/mtl/u_pos/MAMS_contextualized 5 ./data/models/en/mtl/u_pos/MAMS_contextualized --mtl --aux_name upos
```

The second python script which trains all of the models and makes the predictions for the standard datasets is this script:
```
./run_all.sh
```



### Multi task models
#### Conan Doyle
Baseline Span-F1 negation scores for test and validation:

`85.18% and 83.84%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/conan_doyle/negation.jsonnet -s /tmp/any --include-package multitask_negation_target
```


#### SFU (Negation)
Baseline Span-F1 negation scores for test and validation:

`69.20% and 68.09%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/sfu/negation.jsonnet -s /tmp/any --include-package multitask_negation_target
```

#### POS tagging (Streusle data)
Here the task is Universal POS tagging. When running the model once with one Bi-LSTM layer with a CRF decoder the POS tagging accuracy for test and validation respectively is:

`94.3% and 94.26%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/u_pos/pos.jsonnet -s /tmp/any --include-package multitask_negation_target
```

#### Dependency Relation tagging (Streusle data)
Here the task is Dependency Relation tagging where we want to predict the dependency relation tag for a given token but not the dependency graph. When running the model once with one Bi-LSTM layer with a CRF decoder the Dependency Relation tagging accuracy for test and validation respectively is:

`88.22% and 87.49%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/dr/dr.jsonnet -s /tmp/any --include-package multitask_negation_target
```

#### Lexical tagging (Streusle data)
This is a complex BIO tagging task. Accuracy for test and validation respectively is:

`76.94% and 77.94%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/lextag/lextag.jsonnet -s /tmp/any --include-package multitask_negation_target
```

#### SFU (Speculation)
Baseline Span-F1 Spec scores for test and validation:

`42.9 and 56.6`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/sfu_spec/spec.jsonnet -s /tmp/any --include-package multitask_negation_target
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

The speculation corpus has also been filtered so that samples have to contain speculation. This is to better isolate the speculation phenomena, and therefore to better test the models capability on modelling speculative samples. The script below runs the models on this smaller but more isolated speculative data:
``` bash
./scripts/generate_spec_only_predictions.sh
```

The non-filtered speculation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_spec.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_spec.conll)

The filtered speculation data can be found:
1. Laptop dataset: [Validation](./data/main_task/en/laptop/dev_spec_only.conll) and [Test](./data/main_task/en/laptop/test_spec_only.conll)
2. Restaurant dataset: [Validation](./data/main_task/en/restaurant/dev_spec_only.conll) and [Test](./data/main_task/en/restaurant/test_spec_only.conll)

### Predicting on cross domain
**THIS IS NOT IN THE PAPER AND IS ONLY EXPERIMENTAL AT THE MOMENT**

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
**THIS IS NOT IN THE PAPER AND IS ONLY EXPERIMENTAL AT THE MOMENT**
The results created here are Distinct Sentiment (DS) and Strict Text ACcuracy (STAC) using the following script. The results that the script generates for those metrics/subsets are stored in the following JSON file.

``` bash
python scripts/generate_detailed_sentiment_results.py ./data/results/en/ ./data/results/detailed_sentiment_results.json
```

The results are then displayed in the following [notebook.](./notebooks/Multi_task_detailed_sentiment_metrics.ipynb)

### Number of parameters
To find the statistics for the number of parameters in the different models run:
``` bash
python number_parameters.py
```

### Inference time
This test the inference time for the following models after they have been loaded into memory:
1. [STL GloVe](http://ucrel-web.lancs.ac.uk/moorea/research/multitask_negation_for_targeted_sentiment/models/en/stl/laptop/)
2. [STL CWR](http://ucrel-web.lancs.ac.uk/moorea/research/multitask_negation_for_targeted_sentiment/models/en/stl/laptop_contextualized/)
3. [MTL SFU GloVe](http://ucrel-web.lancs.ac.uk/moorea/research/multitask_negation_for_targeted_sentiment/models/en/mtl/sfu/laptop/)
4. [MTL SFU CWR](http://ucrel-web.lancs.ac.uk/moorea/research/multitask_negation_for_targeted_sentiment/models/en/mtl/sfu/laptop_contextualized/)

**NOTE** If you go to any of the model links we use model_0.tar.gz

Both of the models will have been trained on the Laptop dataset. Additionally the links associated to the models above will take you to the location where you can download those models. The inference times will be tested on the Laptop test dataset which contains 800 sentences. Further the models will be tested on the following hardware:
1. GPU - GeForce GTX 1060 6GB
2. CPU - AMD Ryzen 5 1600

And with the following batch sizes:
1. 1
2. 8
3. 16
4. 32

The computer also had 16GB of RAM. Additional the computer will run the model 5 times and time each run and report the minimum and maximum run times. Minimum times are recommended by the [python timeit library](https://docs.python.org/3/library/timeit.html) and maximum is reported to show the potential distribution.

To run these inference time testing run the following:
``` bash
python inference_time.py
```

It will print out a Latex table of results, which when converted to markdown look like the following:

| Embedding | Model | Batch Size | Device | Min Time | Max Time |
| --------- | ----- | ---------- | ------ | -------- | -------- |
| GloVe     | STL   | 1          | CPU    |    10.24 |  10.45   |
| GloVe     | STL   | 8          | CPU    |    7.00  |  7.21    |
| GloVe     | STL   | 16         | CPU    |    6.67  |  6.91    |
| GloVe     | STL   | 32         | CPU    |    6.35  |  6.51    |
| GloVe     | MTL   | 1          | CPU    |    10.06 |  10.26   |
| GloVe     | MTL   | 8          | CPU    |    7.05  |  7.19    |
| GloVe     | MTL   | 16         | CPU    |    6.90  |  6.99    |
| GloVe     | MTL   | 32         | CPU    |    6.41  |  6.46    |
| GloVe     | STL   | 1          | GPU    |    9.24  |  9.26    |
| GloVe     | STL   | 8          | GPU    |    6.58  |  6.67    |
| GloVe     | STL   | 16         | GPU    |    6.34  |  6.36    |
| GloVe     | STL   | 32         | GPU    |    6.12  |  6.26    |
| GloVe     | MTL   | 1          | GPU    |    9.43  |  9.49    |
| GloVe     | MTL   | 8          | GPU    |    6.60  |  6.70    |
| GloVe     | MTL   | 16         | GPU    |    6.26  |  6.55    |
| GloVe     | MTL   | 32         | GPU    |    6.10  |  6.20    |
| CWR       | STL   | 1          | CPU    |    64.79 | 71.26    |
| CWR       | STL   | 8          | CPU    |    43.62 | 49.70    |
| CWR       | STL   | 16         | CPU    |    47.06 | 48.41    |
| CWR       | STL   | 32         | CPU    |    56.76 | 62.77    |
| CWR       | MTL   | 1          | CPU    |    64.01 | 67.90    |
| CWR       | MTL   | 8          | CPU    |    49.05 | 50.00    |
| CWR       | MTL   | 16         | CPU    |    53.74 | 56.42    |
| CWR       | MTL   | 32         | CPU    |    55.33 | 55.79    |
| CWR       | STL   | 1          | GPU    |    23.26 | 23.79    |
| CWR       | STL   | 8          | GPU    |    8.82  | 9.09     |
| CWR       | STL   | 16         | GPU    |    8.57  | 8.86     |
| CWR       | STL   | 32         | GPU    |    8.45  | 9.78     |
| CWR       | MTL   | 1          | GPU    |    23.81 | 23.97    |
| CWR       | MTL   | 8          | GPU    |    9.19  | 9.49     |
| CWR       | MTL   | 16         | GPU    |    8.54  | 8.92     |
| CWR       | MTL   | 32         | GPU    |    8.43  | 8.70     |

Also this data is stored in the following file [./inference_save.json](./inference_save.json)

## Requirements

1. Python >= 3.6.1
2. `pip install -r requirements.txt`
3. `pip install .` 

## Run the tests:

`python -m pytest`
