from typing import Dict, Any, Tuple
import os
import logging

from allennlp.training.trainer import Trainer
from allennlp.training import util as training_util
from allennlp.common.util import dump_metrics
import torch


logger = logging.getLogger(__name__)

def train_one_epoch(trainer: Trainer, epoch_count: int
                    ) -> Tuple[Dict[str, float], Dict[str, float]]:
    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    this_epoch_val_metric: float = None
    metrics: Dict[str, float] = {}

        
    train_metrics = trainer._train_epoch(epoch_count)

    if trainer._validation_data is not None:
        with torch.no_grad():
            # We have a validation set, so compute all the metrics on it.
            val_loss, num_batches = trainer._validation_loss()
            val_metrics = training_util.get_metrics(trainer.model, val_loss, num_batches, reset=True)
            this_epoch_val_metric = val_metrics[trainer._validation_metric]

    for key, value in train_metrics.items():
        metrics["training_" + key] = value
    for key, value in val_metrics.items():
        metrics["validation_" + key] = value

    if trainer._serialization_dir:
        dump_metrics(os.path.join(trainer._serialization_dir, 
                                  f"metrics_epoch_{epoch_count}.json"), metrics)

    # The Scheduler API is agnostic to whether your schedule requires a validation metric -
    # if it doesn't, the validation metric passed here is ignored.
    if trainer._learning_rate_scheduler:
        trainer._learning_rate_scheduler.step(this_epoch_val_metric, epoch_count)
    if trainer._momentum_scheduler:
        trainer._momentum_scheduler.step(this_epoch_val_metric, epoch_count)
    #trainer._save_checkpoint(epoch_count)
    return train_metrics, val_metrics

def multi_task_checkpoint_saver(trainer: Trainer, best_so_far: bool, epoch: int):
    '''
    Different from trainer._save_checkpoint as here we can specify a different 
    trainer `best_so_far` bool value so that the auxiliary tasks best model is 
    not that tasks best model but it is at the epoch of the main task.
    '''
    if trainer._moving_average is not None:
        trainer._moving_average.assign_average_value()

    # These are the training states we need to persist.
    training_states = {
            "metric_tracker": trainer._metric_tracker.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "batch_num_total": trainer._batch_num_total}

    # If we have a learning rate or momentum scheduler, we should persist them too.
    if trainer._learning_rate_scheduler is not None:
        training_states["learning_rate_scheduler"] = trainer._learning_rate_scheduler.state_dict()
    if trainer._momentum_scheduler is not None:
        training_states["momentum_scheduler"] = trainer._momentum_scheduler.state_dict()

    trainer._checkpointer.save_checkpoint(
            model_state=trainer.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=trainer._metric_tracker.is_best_so_far())

    # Restore the original values for parameters so that training will not be affected.
    if trainer._moving_average is not None:
        trainer._moving_average.restore()


def multi_task_training(main_trainer: Trainer, aux_trainer: Trainer) -> Dict[str, Any]:
    '''
    Performs as many epochs as the main task requires and if early stopping 
    is set then it is defined by the main task. The way that multi task is run
    it runs the auxiliary task for one epoch and then the main task for one epoch 
    and then it evaluates the main tasks validation dataset to see if early stopping 
    needs to happen and if so then no more training else it goes for another 
    epoch on auxiliary then main task.

    :returns: Metrics for both auxiliary and main tasks
    '''
    training_util.enable_gradient_clipping(main_trainer.model, 
                                           main_trainer._grad_clipping)
    training_util.enable_gradient_clipping(aux_trainer.model, 
                                           aux_trainer._grad_clipping)
    
    all_metrics: Dict[str, Any] = {}

    for epoch in range(main_trainer._num_epochs):
        aux_train_metrics, aux_val_metrics = train_one_epoch(aux_trainer, epoch)
        main_train_metrics, main_val_metrics = train_one_epoch(main_trainer, epoch)
        # Early stopping if applicable (main task) and tracking the best metric
        main_validation_metric_name = main_trainer._validation_metric
        main_validation_metric = main_val_metrics[main_validation_metric_name]
        main_trainer._metric_tracker.add_metric(main_validation_metric)

        multi_task_checkpoint_saver(aux_trainer, main_trainer._metric_tracker.is_best_so_far(), epoch)
        multi_task_checkpoint_saver(main_trainer, main_trainer._metric_tracker.is_best_so_far(), epoch)

        for task_name, train_metrics in [('aux', aux_train_metrics), 
                                         ('main', main_train_metrics)]:
            for key, value in train_metrics.items():
                all_metrics[f"training_{task_name}_{key}"] = value
        for task_name, val_metrics in [('aux', aux_val_metrics), 
                                       ('main', main_val_metrics)]:
            for key, value in val_metrics.items():
                all_metrics[f"validation_{task_name}_{key}"] = value

        if main_trainer._metric_tracker.should_stop_early():
            logger.info("Ran out of patience.  Stopping training.")
            break
        # Getting the best metrics for the main task
        if main_trainer._metric_tracker.is_best_so_far():
            # Update all the best_ metrics.
            # (Otherwise they just stay the same as they were.)
            all_metrics['best_epoch'] = epoch
            for key, value in main_val_metrics.items():
                all_metrics["best_validation_" + key] = value
            for key, value in aux_val_metrics.items():
                all_metrics["aux_best_validation_" + key] = value
            main_trainer._metric_tracker.best_epoch_metrics = main_val_metrics
    # Load the best model state before returning
    main_best_model_state = main_trainer._checkpointer.best_model_state()
    if main_best_model_state:
        main_trainer.model.load_state_dict(main_best_model_state)
    
    aux_best_model_state = aux_trainer._checkpointer.best_model_state()
    if aux_best_model_state:
        aux_trainer.model.load_state_dict(aux_best_model_state)

    return all_metrics