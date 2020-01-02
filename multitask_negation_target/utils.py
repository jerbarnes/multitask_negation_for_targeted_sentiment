from pathlib import Path
import re

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