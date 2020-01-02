from pathlib import Path
import tempfile
import re

from allennlp.common import from_params, Params
from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import CrfTagger
from allennlp.training.trainer import Trainer
from allennlp.commands.evaluate import evaluate
from allennlp.common.file_utils import cached_path
import torch

from multitask_negation_target.allen.dataset_readers.negation_speculation import NegationSpeculationDatasetReader
from multitask_negation_target import utils

params: Params = Params.from_file(str(Path("./resources/model_configs/targeted_sentiment_baseline.jsonnet").resolve()))

# Download the data unless it is cached
train_url = params['train_data_path']
dev_url = params['validation_data_path'] 
test_url = params['test_data_path'] 
# This just changes it from BIOSE to BIOUL where E=L and S=U
with tempfile.TemporaryDirectory() as temp_data_dir:
    train_fp = Path(cached_path(train_url))
    temp_train_fp = Path(temp_data_dir, 'train.conll')
    utils.from_biose_to_bioul(train_fp, temp_train_fp)
    
    dev_fp = Path(cached_path(dev_url))
    temp_dev_fp = Path(temp_data_dir, 'dev.conll')
    utils.from_biose_to_bioul(dev_fp, temp_dev_fp)
    
    test_fp = Path(cached_path(test_url))
    temp_test_fp = Path(temp_data_dir, 'test.conll')
    utils.from_biose_to_bioul(test_fp, temp_test_fp)

    # Load the datasets
    reader = DatasetReader.from_params(params['dataset_reader'])
    train_instances = reader.read(str(temp_train_fp))
    dev_instances = reader.read(str(temp_dev_fp))
    test_instances = reader.read(str(temp_test_fp))
    all_instances = train_instances + dev_instances + test_instances

    # Create the vocab
    vocab = Vocabulary.from_instances(all_instances)

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    # Model parameters that are taken from the config file
    label_namespace =  params['model'].pop('label_namespace')
    label_encoding = params['model'].pop('label_encoding')
    dropout = params['model'].pop('dropout')
    calculate_span_f1 = params['model'].pop('calculate_span_f1')
    constrain_crf_decoding = params['model'].pop('constrain_crf_decoding')
    include_start_end_transitions = params['model'].pop('include_start_end_transitions')
    text_embedder_params = params['model'].pop('text_field_embedder')
    text_embedder = TextFieldEmbedder.from_params(params=text_embedder_params, vocab=vocab)
    encoder_params = params['model'].pop('encoder')
    encoder = Seq2SeqEncoder.from_params(params=encoder_params)

    sentiment_tagger = CrfTagger(vocab=vocab, text_field_embedder=text_embedder, 
                                 encoder=encoder, label_namespace=label_namespace, 
                                 feedforward=None, label_encoding=label_encoding, 
                                 include_start_end_transitions=include_start_end_transitions, 
                                 constrain_crf_decoding=constrain_crf_decoding, 
                                 calculate_span_f1=calculate_span_f1, dropout=dropout)

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = Trainer.from_params(params=params.pop('trainer'), 
                                      model=sentiment_tagger,
                                      serialization_dir=temp_dir,
                                      iterator=iterator,
                                      validation_data=dev_instances,
                                      train_data=train_instances)
        interesting_metrics = trainer.train()
        best_epoch = interesting_metrics['best_epoch']
        best_validation_span_f1 = interesting_metrics['best_validation_f1-measure-overall']
        best_model_weights = Path(temp_dir, 'best.th')
        best_model_state = torch.load(best_model_weights)
        sentiment_tagger.load_state_dict(best_model_state)

        evaluate_cuda = params['evaluate'].pop("cuda_device")
        test_result = evaluate(sentiment_tagger, test_instances, iterator, 
                               cuda_device=evaluate_cuda, batch_weight_key=None)
        #dev_result = evaluate(negation_tagger, dev_instances, iterator, 
        #                      cuda_device=0, batch_weight_key=None)
        print(f'Best epoch {best_epoch}')
        print(f'Best validation span f1 {best_validation_span_f1}')
        print(f'Test span f1 result {test_result["f1-measure-overall"]}')