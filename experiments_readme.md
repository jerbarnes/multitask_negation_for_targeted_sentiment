# Extended experiments

These experiments are not reported in the paper and are only here to show what else we looked at.

## Auxiliary task results

All of these results show the performance of a model when being trained and evaluated on the selected auxiliary task.

### Conan Doyle

Baseline Span-F1 negation scores for test and validation:

`85.18% and 83.84%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/conan_doyle/negation.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### SFU (Negation)

Baseline Span-F1 negation scores for test and validation:

`69.20% and 68.09%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/sfu/negation.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### SFU (Speculation)

Baseline Span-F1 Spec scores for test and validation:

`42.9 and 56.6`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/sfu_spec/spec.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### UPOS tagging

Here the task is Universal POS tagging. When running the model once with one Bi-LSTM layer with a CRF decoder the POS tagging accuracy for test and validation respectively is:

`94.3% and 94.26%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/u_pos/pos.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Dependency Relation

Here the task is Dependency Relation tagging where we want to predict the dependency relation tag for a given token but not the dependency graph. When running the model once with one Bi-LSTM layer with a CRF decoder the Dependency Relation tagging accuracy for test and validation respectively is:

`88.22% and 87.49%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/dr/dr.jsonnet -s /tmp/any --include-package multitask_negation_target
```

### Lexical analysis

This is a complex BIO tagging task. Accuracy for test and validation respectively is:

`76.94% and 77.94%`

To run that model:
``` bash
allennlp train resources/model_configs/mtl/en/lextag/lextag.jsonnet -s /tmp/any --include-package multitask_negation_target
```

## Cross domain
**THIS IS NOT IN THE PAPER AND IS ONLY EXPERIMENTAL AT THE MOMENT**

This is cross domain between the TSA datasets e.g. training on the MAMS dataset and evaluating on the Restaurant dataset.

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

The Google Colab notebook of the analysis for this can be seen [here.](../notebooks/Cross_Domain.ipynb)

## Detailed sentiment analysis results

**THIS IS NOT IN THE PAPER AND IS ONLY EXPERIMENTAL AT THE MOMENT**

The results created here are Distinct Sentiment (DS) and Strict Text ACcuracy (STAC) using the following script. The results that the script generates for those metrics/subsets are stored in the following JSON file.

``` bash
python scripts/generate_detailed_sentiment_results.py ./data/results/en/ ./data/results/detailed_sentiment_results.json
```

The results are then displayed in the following [notebook.](../notebooks/Multi_task_detailed_sentiment_metrics.ipynb)