# Extended dataset analysis

This contains a few further details on the datasets we used, and to some extent what some of the scripts that analyse these datasets do.

Table of contents:
1. [Negation and Speculation datasets (CD and SFU)](#negation-and-speculation-datasets-(cd-and-sfu))
 * [BIO formatting](#bio-formatting)
 * [Dataset Statistics](#dataset-statistics)
2. [Streusle review corpus](#streusle-review-corpus)
 * [UPOS](#upos)
 * [DR](#dr)
 * [LEX](#lex)


## Negation and Speculation datasets (CD and SFU)

### BIO formatting

For both the Conan Doyle (CD) and SFU review corpus the negation and speculation scopes and cues are dis-continuos such that the BIO scheme does not continue if a cue is within a scope or vice versa, an example of this is shown below where the dis-continus version is example 1 (which is used in this project) and the continuos version is shown in example 2.

|  | Mooner | simply | forgot | to | show | up | for | his | court | appointment | , | if | he | could | ever | remember | , | and |
|--|--------|--------|--------|----|------|----|-----|-----|-------|-------------|---|----|----|-------|------|----------|---|-----|
| Example 1 | O | O | O | O | O | O | O | O | O | O | O | B<sub>*speccue*</sub> | B<sub>*spec*</sub> | B<sub>*speccue*</sub> | B<sub>*spec*</sub> | I<sub>*spec*</sub> | O | O |
| Example 2 | O | O | O | O | O | O | O | O | O | O | O | B<sub>*speccue*</sub> | I<sub>*spec*</sub> | I<sub>*speccue*</sub> | I<sub>*spec*</sub> | I<sub>*spec*</sub> | O | O |

### Dataset Statistics

The table below states the complete dataset statistics (this is the combination of the train, development, and test splits):

| Dataset     | Label/Task       | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
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

#### Statistics broken down into splits

SFU Negation:

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 1,018 (1,749)     | 1,155 (6,562)       | 13,712        | 934                 |
| Dev      | 121 (198)         | 154 (861)           | 1,713         | 114                 |
| Test     | 124 (209)         | 137 (792)           | 1,703         | 117                 |
| Combined | 1,263 (2,156)     | 1,446 (8,215)       | 17,128        | 1,165               |

To generate the data for this table above run `./scripts/negation_split_statistics.sh SFU negation`

----

SFU Speculation

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 390 (425)         | 446 (2,623)         | 13,712        | 309                 |
| Dev      | 58 (63)           | 66 (402)            | 1,713         | 45                  |
| Test     | 65 (74)           | 74 (458)            | 1,703         | 51                  |
| Combined | 513 (562)         | 586 (3,483)         | 17,128        | 405                 |

To generate the data for this table above run `./scripts/negation_split_statistics.sh SFU speculation`

----

Conan Doyle Negation

| Split    | No. Cues (tokens) | No. Scopes (tokens) | No. Sentences | No. Label Sentences |
|----------|-------------------|---------------------|---------------|---------------------|
| Train    | 821 (838)         | 1,507 (6,756)       | 842           | 842                 |
| Dev      | 143 (146)         | 284 (1,283)         | 144           | 144                 |
| Test     | 233 (238)         | 429 (1,722)         | 235           | 235                 |
| Combined | 1,197 (1,222)     | 2,220 (9,761)       | 1,221         | 1,221               |

To generate the data for this table above run `./scripts/negation_split_statistics.sh conandoyle negation`

## Streusle review corpus

For each of the tasks in this dataset the bash commands create vocabularly dataset statistics. When the label is `_` this is converted to the label `NONE` for the Dependency Relation (DR) task and `O` for Lexical analysis (LEX).

This command needs to be ran before any of the following commands:
``` bash
export STANFORDNLP_TEST_HOME=~/stanfordnlp_test
```

### UPOS

Universal Part Of Speech (UPOS)

For U-POS
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_u_pos.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

### DR

Dependency Relations (DR)

``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_dr.jsonnet -s /tmp/dry --include-package multitask_negation_target
```

### LEX

Lexical analysis (LEX)

For LEXTAG
``` bash
allennlp dry-run ./resources/statistic_configs/en/streusle_lextag.jsonnet -s /tmp/dry --include-package multitask_negation_target
```