from pathlib import Path

from flaky import flaky
import pytest
from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model

import multitask_negation_target
from multitask_negation_target.tests import util

#
# For the crf tagger version tests see shared_crf_tagger_test.py
#

class SharedSoftmaxTaggerTest(ModelTestCase):
    DATA_DIR = util.FIXTURES_ROOT / "allen" / "dataset_readers" / "negation_speculation"
    MODEL_DIR = util.FIXTURES_ROOT / "allen" / "models" / "shared_crf_tagger"

    def setUp(self):
        super().setUp()
        self.shared_only_fp = self.MODEL_DIR / "shared_only_softmax_experiment.jsonnet"
        self.shared_model_fp = self.MODEL_DIR / "experiment_shared_encoder_softmax.jsonnet"
        self.shared_skip_connections_model_fp = self.MODEL_DIR / "experiment_shared_encoder_skip_connections_softmax.jsonnet"

        model_fp = self.MODEL_DIR / "experiment_softmax.jsonnet"
        data_fp = self.DATA_DIR / "conan_doyle_data.conllu"
        self.set_up_model(model_fp, data_fp,)

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
    
    def test_shared_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.shared_model_fp)
    
    def test_shared_only_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.shared_only_fp)
    
    def test_shared_and_skip_connections_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.shared_skip_connections_model_fp)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict["tags"]
        assert len(tags) == 2
        assert len(tags[0]) == 9
        assert len(tags[1]) == 18
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="negation_labels")
                assert tag in {"O", "I_scope", "B_scope", "B_cue"}

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 210 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["task_encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
    
    def test_mismatching_shared_encoder_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.shared_model_fp)
        # Make the shared encoder wrong - it should be 210 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["shared_encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
    
    def test_mismatching_task_encoder_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.shared_model_fp)
        # Make the task encoder wrong - it should be 600 to match the shared 
        # encoder
        params["model"]["task_encoder"]["input_size"] = 590
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_task_and_shared_is_required(self):
        params = Params.from_file(self.shared_only_fp)
        del params["model"]["shared_encoder"]
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_shared_cannot_have_skip(self):
        params = Params.from_file(self.shared_only_fp)
        params["model"]["skip_connections"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_mismatching_skip_connections_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.shared_skip_connections_model_fp)
        # Make the task encoder wrong - it should be 810 to match the shared 
        # encoder
        params["model"]["task_encoder"]["input_size"] = 600
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_skip_connections_value(self):
        # If a model does not have a shared encoder and the skip connections is 
        # True should raise a ValueError
        params = Params.from_file(self.param_file)
        params["model"]["skip_connections"]= True
        with pytest.raises(ValueError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))