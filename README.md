# Multitask incorporation of negation for targeted sentiment classification


The idea is to use multitask learning to incorporate negation information into a targeted sentiment classifier

## Model
We could start with the BiLSTM + CRF model I used for sentence-level classification, but adapted to targeted sentiment. Other models could be cool too though.

## Datasets
# Sentiment
1. SemEval 2016 Aspect-based datasets (EN, AR, CH, DU, FR, RU, ES, TU)
2. Multibooked (CA, EU)
3. USAGE (DE)

# Negation
1. EN - ConanDoyleNeg, SFU Review Corpus
2. ES - SFU Spanish
Look for more??


### Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision```
