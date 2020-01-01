import pytest
from allennlp.common.util import ensure_list
from allennlp.common.checks import ConfigurationError


from multitask_negation_target.allen.dataset_readers.negation_speculation import NegationSpeculationDatasetReader
from multitask_negation_target.tests import util

class TestConll2000Reader:
    DATA_DIR = util.FIXTURES_ROOT / "allen" / "dataset_readers" / "negation_speculation"

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool):
        # SFU negation reader
        negation_reader = NegationSpeculationDatasetReader(lazy=lazy, 
                                                           tag_label='negation')
        instances = negation_reader.read(str(self.DATA_DIR / "sfu_negation_speculation_data.conll"))
        instances = ensure_list(instances)
        assert len(instances) == 2

        sfu_expected_tokens_1 = ['I', 'bought', 'and', 'read', '"', 'A', 'Painted', 
                                 'House', '"', 'as', 'soon', 'as', 'it', 'came', 
                                 'out', 'as', 'I', 'do', 'with', 'all', 'John', 
                                 'Grisham', 'books', '.']
        sfu_expected_negation_1 = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                   'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                   'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                   'O', 'O', 'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == sfu_expected_tokens_1
        assert fields["tags"].labels == sfu_expected_negation_1

        sfu_expected_tokens_2 = ['I', 'was', 'interested', 'to', 'read', 'how', 
                                 'he', 'treated', 'a', 'story', 'that', 'did', 
                                 'not', 'revolve', 'around', 'a', 'legal', 
                                 'case', '.']
        sfu_expected_negation_2 = ['B_negcue', 'I_negcue', 'O', 'O', 'O', 'O', 
                                   'O', 'O', 'O', 'O', 'B_neg', 'O', 
                                   'B_negcue', 'B_neg', 'I_neg', 'I_neg', 'I_neg', 
                                   'I_neg', 'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == sfu_expected_tokens_2
        assert fields["tags"].labels == sfu_expected_negation_2

        # SFU speculation reader
        speculation_reader = NegationSpeculationDatasetReader(lazy=lazy, 
                                                              tag_label='speculation')
        instances = speculation_reader.read(str(self.DATA_DIR / "sfu_negation_speculation_data.conll"))
        instances = ensure_list(instances)
        assert len(instances) == 2

        sfu_expected_speculation_1 = ['B_spec', 'O', 'O', 'O', 'O', 'O', 'O', 
                                      'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                      'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                                      'O', 'O', 'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == sfu_expected_tokens_1
        assert fields["tags"].labels == sfu_expected_speculation_1

        sfu_expected_speculation_2 = ['O', 'O', 'B_spec', 'I_spec', 'B_speccue', 
                                      'O', 'B_speccue', 'I_speccue', 'O', 
                                      'B_spec', 'O', 'O', 'O', 'O', 'O', 'O', 
                                      'O', 'O', 'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == sfu_expected_tokens_2
        assert fields["tags"].labels == sfu_expected_speculation_2

        # Conan Doyle negation reader which does not have a speculation column
        negation_reader = NegationSpeculationDatasetReader(lazy=lazy, 
                                                           tag_label='negation')
        instances = negation_reader.read(str(self.DATA_DIR / "conan_doyle_data.conllu"))
        instances = ensure_list(instances)
        assert len(instances) == 2

        conan_expected_tokens_1 = ['No', 'woman', 'would', 'ever', 'send', 'a', 
                                   'reply-paid', 'telegram', '.']
        conan_expected_negation_1 = ['B_cue', 'B_scope', 'I_scope', 'I_scope', 
                                     'I_scope', 'I_scope', 'I_scope', 
                                     'I_scope', 'O']

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == conan_expected_tokens_1
        assert fields["tags"].labels == conan_expected_negation_1

        conan_expected_tokens_2 = ['``', 'I', 'have', 'had', 'a', 'most', 
                                   'singular', 'and', 'unpleasant', 'experience', 
                                   ',', 'Mr.', 'Holmes', ',', "''", 'said', 
                                   'he', '.']
        conan_expected_negation_2 = ['O', 'B_scope', 'I_scope', 'I_scope', 
                                     'I_scope', 'I_scope', 'O', 'O', 'B_scope', 
                                     'I_scope', 'O', 'O', 'O', 'O', 'O', 'O', 
                                     'O', 'O']

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == conan_expected_tokens_2
        assert fields["tags"].labels == conan_expected_negation_2

        # Ensure it raises an error if the tag_label is not `negation` or 
        # `speculation`
        with pytest.raises(ConfigurationError):
            NegationSpeculationDatasetReader(lazy=lazy, tag_label='negatio')
        