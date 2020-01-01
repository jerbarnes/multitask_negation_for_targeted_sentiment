from pathlib import Path
import tempfile

from allennlp.common import from_params, Params
from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import CrfTagger
from allennlp.training.trainer import Trainer
from allennlp.commands.evaluate import evaluate
import torch

from multitask_negation_target.allen.dataset_readers.negation_speculation import NegationSpeculationDatasetReader

params: Params = Params.from_file(str(Path("./resources/model_configs/test.jsonnet").resolve()))
# Load the datasets
reader = DatasetReader.from_params(params['dataset_reader'])
train_instances = reader.read(params['train_data_path'])
dev_instances = reader.read(params['validation_data_path'])
test_instances = reader.read(params['test_data_path'])
all_instances = train_instances + dev_instances + test_instances

# Create the vocab
vocab = Vocabulary.from_instances(all_instances)

iterator = DataIterator.from_params(params.pop("iterator"))
iterator.index_with(vocab)

# Model parameters that are taken from the config file
label_namespace =  params['model'].pop('label_namespace')
label_encoding = 'BIO'
dropout = params['model'].pop('dropout')
calculate_span_f1 = params['model'].pop('calculate_span_f1')
constrain_crf_decoding = params['model'].pop('constrain_crf_decoding')
include_start_end_transitions = params['model'].pop('include_start_end_transitions')
text_embedder_params = params['model'].pop('text_field_embedder')
text_embedder = TextFieldEmbedder.from_params(params=text_embedder_params, vocab=vocab)
encoder_params = params['model'].pop('encoder')
encoder = Seq2SeqEncoder.from_params(params=encoder_params)

negation_tagger = CrfTagger(vocab=vocab, text_field_embedder=text_embedder, 
                            encoder=encoder, label_namespace=label_namespace, 
                            feedforward=None, label_encoding=label_encoding, 
                            include_start_end_transitions=include_start_end_transitions, 
                            constrain_crf_decoding=constrain_crf_decoding, 
                            calculate_span_f1=calculate_span_f1, dropout=dropout)

with tempfile.TemporaryDirectory() as temp_dir:
    trainer = Trainer.from_params(params=params.pop('trainer'), 
                                  model=negation_tagger,
                                  serialization_dir=temp_dir,
                                  iterator=iterator,
                                  validation_data=dev_instances,
                                  train_data=train_instances)
    interesting_metrics = trainer.train()
    best_epoch = interesting_metrics['best_epoch']
    best_validation_span_f1 = interesting_metrics['best_validation_f1-measure-overall']
    best_model_weights = Path(temp_dir, 'best.th')
    best_model_state = torch.load(best_model_weights)
    negation_tagger.load_state_dict(best_model_state)

    evaluate_cuda = params['evaluate'].pop("cuda_device")
    test_result = evaluate(negation_tagger, test_instances, iterator, 
                           cuda_device=evaluate_cuda, batch_weight_key=None)
    #dev_result = evaluate(negation_tagger, dev_instances, iterator, 
    #                      cuda_device=0, batch_weight_key=None)
    print(f'Best epoch {best_epoch} best validation span f1 {best_validation_span_f1}')
    print(f'Test span f1 result {test_result["f1-measure-overall"]}')