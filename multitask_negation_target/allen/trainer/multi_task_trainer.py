import logging
import datetime
import time
from typing import Optional, Dict, List, Any
from pathlib import Path
import itertools
import tempfile
import shutil

from allennlp.training import util as training_util
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.commands.evaluate import evaluate
from allennlp.training.trainer_base import TrainerBase
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.trainer import Trainer

from multitask_negation_target.allen.models.shared_crf_tagger import SharedCrfTagger
from multitask_negation_target.allen.utils import multi_task_training

logger = logging.getLogger(__name__)

@TrainerBase.register("multi_task_trainer")
class MultiTaskTrainer(TrainerBase):
    '''
    Model saving. All models are saved in their serialization format within the
    given `serialization_dir`. Each task is saved as `{task name}_model.tar.gz`
    apart from the main task which is saved as `model.tar.gz`.
    '''
    def __init__(self, params: Params,  serialization_dir: str, 
                 recover: bool = False, cache_directory: Optional[str] = None, 
                 cache_prefix: Optional[str] = None):
        self.original_params = params.duplicate()
        self.main_serialization_dir = serialization_dir
        if recover or cache_directory or cache_prefix:
            raise NotImplementedError(f'Currently do not support `recover` {recover}, '
                                      f'`cache_directory` {cache_directory}, or '
                                      f'`cache_prefix` {cache_prefix}')
        task_order = params.get('trainer').pop('task_order')
        main_task = params.get('trainer').pop('main_task')
        self.task_order = task_order
        if main_task != task_order[-1]: 
            raise ConfigurationError(f'main task {main_task} with `trainer` has'
                                     ' to be equal to the last task in the '
                                     f'`task_order` {task_order}')
        logger.warning(f"Main task {main_task}")
        logger.warning("Training tasks each epoch in the following order:")
        for task_index, task in enumerate(task_order):
            logger.warning(f'{task_index}: {task}')

        # Get shared iterator
        shared_values = params.pop('shared_values')
        iterator = DataIterator.from_params(shared_values.pop("iterator"))
        self.iterator = iterator
        # Get dataset information
        # task name: dataset split name: data
        all_task_data: Dict[str, Dict[str, List[Instance]]] = {}
        self.task_params = {} 
        for task in task_order:
            logger.warning(f'Loading dataset for {task}')
            task_data = {}
            task_params = params.get(task)
            self.task_params[task] = task_params.duplicate()
            
            dataset_reader = DatasetReader.from_params(task_params.pop('dataset_reader'))
            
            task_data['train'] = dataset_reader.read(task_params.pop('train_data_path'))
            task_data['validation'] = dataset_reader.read(task_params.pop('validation_data_path'))
            task_data['test'] = dataset_reader.read(task_params.pop('test_data_path'))
            all_task_data[task] = task_data
        
        # Create the vocab
        logger.warning('Creating Vocab from all task data')
        all_instances = [data for task_data in all_task_data.values() 
                              for data in task_data.values()]
        all_instances = list(itertools.chain.from_iterable(all_instances))
        vocab = Vocabulary.from_instances(all_instances)
        iterator.index_with(vocab)
        logger.warning('Iterator indexed')

        # Shared model parameters
        text_embedder = None
        if 'text_field_embedder' in shared_values:
            logger.warning('Creating shared text embedder')
            text_embedder_params = shared_values.pop('text_field_embedder')
            text_embedder = TextFieldEmbedder.from_params(params=text_embedder_params, 
                                                          vocab=vocab)
        shared_encoder = None
        if 'shared_encoder' in shared_values:
            logger.warning('Creating shared Sequence Encoder')
            shared_encoder_params = shared_values.pop('shared_encoder')
            shared_encoder = Seq2SeqEncoder.from_params(params=shared_encoder_params)
        # Creating task specific models
        task_models: Dict[str, SharedCrfTagger] = {}
        for task in task_order:
            logger.warning(f'Creating shared model for task {task}')
            task_params = params.get(task)
            task_model_params = task_params.pop('model')
            task_text_embedder = None
            if text_embedder is not None:
                task_text_embedder = text_embedder

            if task_text_embedder is not None and "text_field_embedder" in task_model_params:
                raise ConfigurationError('Cannot have a shared text field '
                                         'embedder and a task specific one')
            if shared_encoder is not None and "shared_encoder" in task_model_params:
                raise ConfigurationError('Cannot have a shared encoder in shared_values '
                                         'and a task specific shared encoder')
            
            if "text_field_embedder" in task_model_params:
                task_text_embedder_params = task_model_params.pop('text_field_embedder')
                task_text_embedder = TextFieldEmbedder.from_params(params=task_text_embedder_params, 
                                                                   vocab=vocab)

            if task_model_params.pop('type') != 'shared_crf_tagger':
                raise ConfigurationError('The SharedCRF tagger model is the '
                                         f'only supported model. Error task {task}')
            task_models[task] = SharedCrfTagger.from_params(vocab=vocab, 
                                                            text_field_embedder=task_text_embedder, 
                                                            shared_encoder=shared_encoder,
                                                            params=task_model_params)
        # Task specific trainers
        task_trainers: Dict[str, Trainer] = {}
        self.task_serialization_dir = {}
        for task in task_order:
            logger.warning(f'Creating {task} trainer')
            task_serialization_dir = str(Path(serialization_dir, task))
            self.task_serialization_dir[task] = task_serialization_dir
            logger.warning(f'Task {task} serialization directory: {task_serialization_dir}')

            task_trainer_params = params.get(task).pop('trainer')
            task_train_data = all_task_data[task]['train']
            task_validation_data = all_task_data[task]['validation']
            task_model = task_models[task]
            task_trainers[task] = Trainer.from_params(params=task_trainer_params, 
                                                      model=task_model,
                                                      serialization_dir=task_serialization_dir,
                                                      iterator=iterator,
                                                      validation_data=task_validation_data,
                                                      train_data=task_train_data)
        # Getting task specific evaluation data 
        self.task_cuda_evaluation = {}
        self.auxiliary_task_validation_data = {}
        self.all_task_test_data = {}
        for task in task_order:
            # If not setting for cuda or not then cuda is assumed.
            self.task_cuda_evaluation[task] = 0
            if 'evaluate' in params.get(task):
                if 'cuda_device' in params.get(task).get('evaluate'):
                    is_cuda = params.get(task).pop('evaluate')['cuda_device']
                    self.task_cuda_evaluation[task] = is_cuda 
            if task != main_task:
                self.auxiliary_task_validation_data[task] = all_task_data[task]['validation']
            self.all_task_test_data[task] = all_task_data[task]['test']
        # Remove all of the tasks from the params
        for task in task_order:
            params.pop(task)
        params.pop('trainer')
        params.assert_empty('MultiTaskTrainer')
        self.task_trainers = task_trainers

    def train(self) -> Dict[str, Any]:
        '''
        :returns: A dictionary containing validation and test results on all of 
                  the tasks.
        '''
        main_task_name = self.task_order[-1]
        main_task_trainer = self.task_trainers[main_task_name]
        main_trainer_name = (main_task_trainer, main_task_name)

        auxiliary_task_names = self.task_order[:-1]
        auxiliary_task_trainers = [self.task_trainers[task_name] 
                                   for task_name in auxiliary_task_names]
        auxiliary_trainers_names = (auxiliary_task_trainers, auxiliary_task_names)
        training_start_time = time.time()
        # Training
        training_metrics = multi_task_training(main_trainer_name, auxiliary_trainers_names)
        training_elapsed_time = time.time() - training_start_time
        training_metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
        
        all_metrics = {**training_metrics}
        logger.info('Evaluating the Auxiliary tasks on their validation and test data')
        
        self.aux_models: List[SharedCrfTagger] = []
        self.all_models = {}
        for aux_trainer, aux_name in zip(*auxiliary_trainers_names):
            logger.info(f'Evaluating {aux_name} on their validation data')
            validation_instances = self.auxiliary_task_validation_data[aux_name]
            cuda_device = self.task_cuda_evaluation[aux_name]
            results = evaluate(aux_trainer.model, validation_instances, 
                               self.iterator, cuda_device=cuda_device, 
                               batch_weight_key=None)
            for key, value in results.items():
                all_metrics[f"aux_{aux_name}_best_validation_{key}"] = value

            logger.info(f'Evaluating {aux_name} on their test data')
            test_instances = self.all_task_test_data[aux_name]
            results = evaluate(aux_trainer.model, test_instances, 
                               self.iterator, cuda_device=cuda_device, 
                               batch_weight_key=None)
            for key, value in results.items():
                all_metrics[f"aux_{aux_name}_best_test_{key}"] = value
            self.aux_models.append(aux_trainer.model)
            self.all_models[aux_name] = aux_trainer.model
        self.all_models[self.task_order[-1]] = main_task_trainer.model

        logger.info(f'Evaluating the main task {main_task_name} on their test data')
        test_instances = self.all_task_test_data[main_task_name]
        cuda_device = self.task_cuda_evaluation[main_task_name]
        results = evaluate(main_task_trainer.model, test_instances, 
                           self.iterator, cuda_device=cuda_device, 
                           batch_weight_key=None)
        for key, value in results.items():
            all_metrics[f"test_{key}"] = value

        # Saving the models in serialization format
        for task in self.task_order:
            serialization_dir = self.task_serialization_dir[task]
            best_weights_fp = str(Path(serialization_dir, 'best.th').resolve())
            task_params = self.task_params[task]
            del task_params['evaluate']
            task_params['model']['text_field_embedder'] = self.original_params['shared_values']['text_field_embedder'].duplicate().as_ordered_dict()
            task_params['model']['shared_encoder'] = self.original_params['shared_values']['shared_encoder'].duplicate().as_ordered_dict()
            task_params['iterator'] = self.original_params['shared_values']['iterator'].duplicate().as_ordered_dict()
            with tempfile.TemporaryDirectory() as temp_dir:
                task_params.to_file(str(Path(temp_dir, CONFIG_NAME).resolve()))
                self.all_models[task].vocab.save_to_files(str(Path(temp_dir, "vocabulary").resolve()))
                archive_model(temp_dir, weights=best_weights_fp)
                saved_model_fp = str(Path(temp_dir, 'model.tar.gz').resolve())
                new_saved_model_fp = Path(self.main_serialization_dir, f'{task}_model.tar.gz')
                # main task does not require name of the task
                if task == self.task_order[-1]:
                    new_saved_model_fp = Path(self.main_serialization_dir, 'model.tar.gz')
                new_saved_model_fp = str(new_saved_model_fp.resolve())
                shutil.move(saved_model_fp, new_saved_model_fp)
        self.model = main_task_trainer.model
        return all_metrics

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        return cls(params, serialization_dir, recover, cache_directory, cache_prefix)