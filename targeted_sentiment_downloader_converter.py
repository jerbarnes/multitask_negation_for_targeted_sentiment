from pathlib import Path
import tempfile

from allennlp.common.file_utils import cached_path
from target_extraction.dataset_parsers import multi_aspect_multi_sentiment_atsa
from target_extraction.tokenizers import spacy_tokenizer

from multitask_negation_target import utils

laptop_train_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/laptop14_train.txt"
laptop_dev_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/laptop14_dev.txt"
laptop_test_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/laptop14_test.txt"

laptop_urls = [laptop_train_url, laptop_dev_url, laptop_test_url]

restaurant_train_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/rest_total_train.txt"
restaurant_dev_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/rest_total_dev.txt"
restaurant_test_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/b2491be09133e9331a286c9d826d4c5b8c0a6671/data_conll/rest_total_test.txt"

restaurant_urls = [restaurant_train_url, restaurant_dev_url, restaurant_test_url]

sentiment_data_dir = Path('.', 'data', 'main_task', 'en')
laptop_data_dir = Path(sentiment_data_dir, 'laptop')
restaurant_data_dir = Path(sentiment_data_dir, 'restaurant')

common_file_names = ['train.conll', 'dev.conll', 'test.conll']
data_dir_urls = [(restaurant_data_dir, restaurant_urls), 
                 (laptop_data_dir, laptop_urls)]
for data_dir, urls in data_dir_urls:
    for url, file_name in zip(urls, common_file_names):
        downloaded_fp = cached_path(url)
        new_fp = Path(data_dir, file_name)
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        utils.from_biose_to_bioul(Path(downloaded_fp), new_fp)

mams_data_dir = Path(sentiment_data_dir, 'MAMS')
mams_data_dir.mkdir(parents=True, exist_ok=True)
split_names = ['train', 'val', 'test']
for split_name, file_name in zip(split_names, common_file_names):
    if split_name == 'train':
        collection = multi_aspect_multi_sentiment_atsa(split_name, original=False)
    else:
        collection = multi_aspect_multi_sentiment_atsa(split_name)
    collection.tokenize(spacy_tokenizer())
    collection.sequence_labels(label_key='target_sentiments')
    conll_fp = Path(mams_data_dir, file_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fp = Path(temp_dir, 'temp_file.conll')
        collection.to_conll_file(temp_fp, gold_label_key='sequence_labels')
        utils.from_bio_to_bioul(temp_fp, conll_fp)
