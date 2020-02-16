from pathlib import Path
import re

from allennlp.data.dataset_readers.dataset_utils import to_bioul

def from_bio_to_bioul(bio_fp: Path, bioul_fp: Path) -> None:
    '''
    :NOTE: This also removes lines that start with `#` and changes the 
           Sentiment labels with the following dictionary:
           `{'positive': 'POS', 'neutral': 'NEU', 'negative': 'NEG'}`

    :param bio_fp: File path to the data that is in CONLL like format: 
                   TOKEN LABEL\n where sentences are split by empty new lines.
                   The label format is in BIO = Beginning of, inside of, 
                   outside.
    :param bioul_fp: File path to save the data that is in `bio_fp` to 
                     this file but in BIOUL
    '''
    sentiment_tag_convert = {'positive': 'POS', 'neutral': 'NEU', 'negative': 'NEG'}
    with bioul_fp.open('w+') as bioul_file:
        with bio_fp.open('r') as bio_file:
            tokens = []
            labels = []
            for line in bio_file:
                if not line.strip():
                    labels = to_bioul(labels, encoding='BIO')
                    temp_labels = []
                    for label in labels:
                        if len(label.split('-')) == 1:
                            temp_labels.append(label)
                        else:
                            bio_tag, sentiment_tag = label.split('-')
                            sentiment_tag = sentiment_tag_convert[sentiment_tag]
                            temp_labels.append(f'{bio_tag}-{sentiment_tag}')
                    labels = temp_labels
                    for token, label in zip(tokens, labels):
                        bioul_file.write(f'{token} {label}\n')
                    bioul_file.write('\n')
                    tokens = []
                    labels = []
                else:
                    if re.search(r'^#', line):
                        continue
                    token, label = line.split()
                    tokens.append(token)
                    labels.append(label)
            if tokens:
                labels = to_bioul(labels, encoding='BIO')
                temp_labels = []
                for label in labels:
                    if len(label.split('-')) == 1:
                        temp_labels.append(label)
                    else:
                        bio_tag, sentiment_tag = label.split('-')
                        sentiment_tag = sentiment_tag_convert[sentiment_tag]
                        temp_labels.append(f'{bio_tag}-{sentiment_tag}')
                labels = temp_labels
                for token, label in zip(tokens, labels):
                    bioul_file.write(f'{token} {label}\n')

def from_biose_to_bioul(biose_fp: Path, bioul_fp: Path) -> None:
    '''
    :param biose_fp: File path to the data that is in CONLL like format: 
                     TOKEN LABEL\n where sentences are split by empty new lines.
                     The label format is in BIOSE = Beginning of, inside of, 
                     outside, single unit, and end of.
    :param bioul_fp: File path to save the data that is in `biose_fp` to 
                     this file but in BIOUL format which is idenitcal to 
                     BIOSE but where S=U and E=L.
    '''
    with bioul_fp.open('w+') as bioul_file:
        with biose_fp.open('r') as biose_file:
            for line in biose_file:
                if not line.strip():
                    bioul_file.write(line)
                else:
                    token, label = line.split()
                    if re.search('^E', label):
                        label = re.sub('^E', 'L', label)
                    elif re.search('^S', label):
                        label = re.sub('^S', 'U', label)
                    line = f'{token} {label}\n'
                    bioul_file.write(line)

def from_biose_to_bio(biose_fp: Path, bio_fp: Path) -> None:
    '''
    :param biose_fp: File path to the data that is in CONLL like format: 
                     TOKEN LABEL\n where sentences are split by empty new lines.
                     The label format is in BIOSE = Beginning of, inside of, 
                     outside, single unit, and end of.
    :param bio_fp: File path to save the data that is in `biose_fp` to 
                   this file but in BIO format where S tags will become B tags 
                   and E tags will become I tags.
    '''
    with bio_fp.open('w+') as bio_file:
        with biose_fp.open('r') as biose_file:
            for line in biose_file:
                if not line.strip():
                    bio_file.write(line)
                else:
                    token, label = line.split()
                    if re.search('^E', label):
                        label = re.sub('^E', 'I', label)
                    elif re.search('^S', label):
                        label = re.sub('^S', 'B', label)
                    line = f'{token} {label}\n'
                    bio_file.write(line)