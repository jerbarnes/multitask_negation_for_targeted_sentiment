from pathlib import Path

from multitask_negation_target import utils

from allennlp.common.file_utils import cached_path

laptop_train_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_train.txt"
laptop_dev_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_dev.txt"
laptop_test_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/laptop14_test.txt"

laptop_urls = [laptop_train_url, laptop_dev_url, laptop_test_url]

restaurant_train_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/rest_total_train.txt"
restaurant_dev_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/rest_total_dev.txt"
restaurant_test_url = "https://raw.githubusercontent.com/lixin4ever/E2E-TBSA/master/data_conll/rest_total_test.txt"

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