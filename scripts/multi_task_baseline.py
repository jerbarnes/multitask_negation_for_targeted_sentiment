from pathlib import Path
import tempfile
import re
import copy

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
from multitask_negation_target import allen
from multitask_negation_target.allen.models.shared_crf_tagger import SharedCrfTagger

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_fp", type=parse_path,
                        help='File Path to sentiment model configuration')
    args = parser.parse_args()
    params: Params = Params.from_file(str(args.model_config_fp))

    # Negation data
    negation_reader = DatasetReader.from_params(params['negation_dataset_reader'])
    negation_train_instances = negation_reader.read(params['negation_train_data_path'])
    negation_dev_instances = negation_reader.read(params['negation_validation_data_path'])
    negation_test_instances = negation_reader.read(params['negation_test_data_path'])

    # Targeted Sentiment data
    # Download the data unless it is cached
    sentiment_train_url = params['target_sentiment_train_data_path']
    sentiment_dev_url = params['target_sentiment_validation_data_path'] 
    sentiment_test_url = params['target_sentiment_test_data_path'] 
    # This just changes it from BIOSE to BIOUL where E=L and S=U
    with tempfile.TemporaryDirectory() as temp_data_dir:
        train_fp = Path(cached_path(sentiment_train_url))
        sentiment_temp_train_fp = Path(temp_data_dir, 'train.conll')
        utils.from_biose_to_bioul(train_fp, sentiment_temp_train_fp)
        
        dev_fp = Path(cached_path(sentiment_dev_url))
        sentiment_temp_dev_fp = Path(temp_data_dir, 'dev.conll')
        utils.from_biose_to_bioul(dev_fp, sentiment_temp_dev_fp)
        
        test_fp = Path(cached_path(sentiment_test_url))
        sentiment_temp_test_fp = Path(temp_data_dir, 'test.conll')
        utils.from_biose_to_bioul(test_fp, sentiment_temp_test_fp)

        # Load the datasets
        sentiment_reader = DatasetReader.from_params(params['target_sentiment_dataset_reader'])
        sentiment_train_instances = sentiment_reader.read(str(sentiment_temp_train_fp))
        sentiment_dev_instances = sentiment_reader.read(str(sentiment_temp_dev_fp))
        sentiment_test_instances = sentiment_reader.read(str(sentiment_temp_test_fp))
        all_instances = (sentiment_train_instances + sentiment_dev_instances + 
                        sentiment_test_instances + negation_train_instances + 
                        negation_dev_instances + negation_test_instances)

        # Create the vocab
        vocab = Vocabulary.from_instances(all_instances)

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(vocab)

        # shared model parameters
        dropout = params['shared_model'].pop('dropout')
        calculate_span_f1 = params['shared_model'].pop('calculate_span_f1')
        constrain_crf_decoding = params['shared_model'].pop('constrain_crf_decoding')
        include_start_end_transitions = params['shared_model'].pop('include_start_end_transitions')
        text_embedder_params = params['shared_model'].pop('text_field_embedder')
        text_embedder = TextFieldEmbedder.from_params(params=text_embedder_params, vocab=vocab)
        encoder_params = params['shared_model'].pop('shared_encoder')
        shared_encoder = Seq2SeqEncoder.from_params(params=encoder_params)
        trainer = params.pop('trainer')
        negation_trainer = copy.deepcopy(trainer)
        sentiment_trainer = copy.deepcopy(trainer)
        evaluate_cuda = params['evaluate'].pop("cuda_device")

        # Negation parameters
        negation_label_namespace =  params['negation_model'].pop('label_namespace')
        negation_label_encoding = params['negation_model'].pop('label_encoding')
        negation_encoder_params = params['negation_model'].pop('task_encoder')
        negation_encoder = Seq2SeqEncoder.from_params(params=negation_encoder_params)

        # Negation model
        negation_tagger = SharedCrfTagger(vocab=vocab, text_field_embedder=text_embedder, 
                                    shared_encoder=shared_encoder, label_namespace=negation_label_namespace, 
                                    feedforward=None, label_encoding=negation_label_encoding, 
                                    include_start_end_transitions=include_start_end_transitions, 
                                    constrain_crf_decoding=constrain_crf_decoding, 
                                    calculate_span_f1=calculate_span_f1, dropout=dropout,
                                    task_encoder=negation_encoder, skip_connections=True)
        # Sentiment parameters
        sentiment_label_namespace =  params['sentiment_model'].pop('label_namespace')
        sentiment_label_encoding = params['sentiment_model'].pop('label_encoding')
        sentiment_encoder_params = params['sentiment_model'].pop('task_encoder')
        sentiment_encoder = Seq2SeqEncoder.from_params(params=sentiment_encoder_params)

        # Sentiment model
        sentiment_tagger = SharedCrfTagger(vocab=vocab, text_field_embedder=text_embedder, 
                                    shared_encoder=shared_encoder, label_namespace=sentiment_label_namespace, 
                                    feedforward=None, label_encoding=sentiment_label_encoding, 
                                    include_start_end_transitions=include_start_end_transitions, 
                                    constrain_crf_decoding=constrain_crf_decoding, 
                                    calculate_span_f1=calculate_span_f1, dropout=dropout,
                                    task_encoder=sentiment_encoder, skip_connections=True)
        # Train Negation and Sentiment model in a multi task fashion
        print('Training Negation and Sentiment model')
        with tempfile.TemporaryDirectory() as negation_temp_dir:
            with tempfile.TemporaryDirectory() as sentiment_temp_dir:
                negation_trainer = Trainer.from_params(params=negation_trainer, 
                                                       model=negation_tagger,
                                                       serialization_dir=negation_temp_dir,
                                                       iterator=iterator,
                                                       validation_data=negation_dev_instances,
                                                       train_data=negation_train_instances)
                sentiment_trainer = Trainer.from_params(params=sentiment_trainer, 
                                                        model=sentiment_tagger,
                                                        serialization_dir=sentiment_temp_dir,
                                                        iterator=iterator,
                                                        validation_data=sentiment_dev_instances,
                                                        train_data=sentiment_train_instances)
                interesting_metrics = allen.utils.multi_task_training(main_trainer=sentiment_trainer, 
                                                                      aux_trainer=negation_trainer)
                
                negation_best_model_weights = Path(negation_temp_dir, 'best.th')
                negation_best_model_state = torch.load(negation_best_model_weights)
                negation_tagger.load_state_dict(negation_best_model_state)
                
                negation_dev_results = evaluate(negation_tagger, negation_dev_instances, iterator, 
                                                cuda_device=evaluate_cuda, batch_weight_key=None)
                negation_test_results = evaluate(negation_tagger, negation_test_instances, iterator, 
                                                cuda_device=evaluate_cuda, batch_weight_key=None)
                
                sentiment_best_model_weights = Path(sentiment_temp_dir, 'best.th')
                sentiment_best_model_state = torch.load(sentiment_best_model_weights)
                sentiment_tagger.load_state_dict(sentiment_best_model_state)
                sentiment_test_results = evaluate(sentiment_tagger, sentiment_test_instances, iterator, 
                                                  cuda_device=evaluate_cuda, batch_weight_key=None)

                print(f'Best epoch {interesting_metrics["best_epoch"]}')
                print('Negation Results')
                print(f'Validation F1 measure: {negation_dev_results["f1-measure-overall"]}')
                print(f'Test F1 measure: {negation_test_results["f1-measure-overall"]}')
                print('Sentiment Results')
                print(f'Validation F1 measure: {interesting_metrics["best_validation_f1-measure-overall"]}')
                print(f'Test F1 measure: {sentiment_test_results["f1-measure-overall"]}')