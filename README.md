# Multitask incorporation of negation for targeted sentiment classification


The idea is to use multitask learning to incorporate negation information into a targeted sentiment classifier

## Model
We could start with the BiLSTM + CRF model I used for sentence-level classification, but adapted to targeted sentiment. Other models could be cool too though.

## Datasets
# Sentiment
1. SemEval 2016 Aspect-based datasets (EN, ES)

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


### Requirements

1. Python >= 3.6
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision```