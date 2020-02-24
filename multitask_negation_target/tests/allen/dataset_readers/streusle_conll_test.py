import pytest
from allennlp.common.util import ensure_list
from allennlp.common.checks import ConfigurationError


from multitask_negation_target.allen.dataset_readers.streusle_conll import StreusleDatasetReader
from multitask_negation_target.tests import util

class TestStreusle:
    DATA_DIR = util.FIXTURES_ROOT / "allen" / "dataset_readers" / "streusle"

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool):
        tag_2_index = ['UPOS', 'XPOS', 'DR', 'SMWE', 'SS']
        # SFU negation reader
        

        expected_tokens_1 = ['Buyer', 'Beware', '!!']
        expected_upos_1 = ['NOUN', 'VERB', 'PUNCT']
        expected_xpos_1 = ['NN', 'VB', '.']
        expected_dr_1 = ['nsubj', 'root', 'punct']
        expected_smwe_1 = ['FALSE', 'FALSE', 'FALSE']
        expected_ss_1 = ['n.PERSON', 'v.cognition', 'NONE']
        tags_1 = [expected_upos_1, expected_xpos_1, expected_dr_1, 
                  expected_smwe_1, expected_ss_1]

        expected_tokens_2 = ['Rusted', 'out', 'and', 'unsafe', 'cars', 'sold', 
                             'here', '!']
        expected_upos_2 = ['VERB', 'ADP', 'CCONJ', 'ADJ', 'NOUN', 'VERB', 
                           'ADV', 'PUNCT']
        expected_xpos_2 = ['VBN', 'RP', 'CC', 'JJ', 'NNS', 'VBN', 'RB', '.']
        expected_dr_2 = ['amod', 'compound', 'cc', 'conj', 'nsubj:pass', 
                         'root', 'advmod', 'punct']
        expected_smwe_2 = ['TRUE', 'TRUE', 'FALSE', 'FALSE', 'FALSE', 
                           'FALSE', 'FALSE', 'FALSE']
        expected_ss_2 = ['v.stative', 'NONE', 'NONE', 'NONE', 'n.ARTIFACT', 
                         'v.possession', 'NONE', 'NONE']
        tags_2 = [expected_upos_2, expected_xpos_2, expected_dr_2, 
                  expected_smwe_2, expected_ss_2]

        expected_tokens_3 = ['Have', 'a', 'real', 'mechanic', 'check', 'before', 
                             'you', 'buy', '!!!!']
        expected_upos_3 = ['VERB', 'DET', 'ADJ', 'NOUN', 'VERB', 'SCONJ', 
                           'PRON', 'VERB', 'PUNCT']
        expected_xpos_3 = ['VB', 'DT', 'JJ', 'NN', 'VB', 'IN', 'PRP', 'VBP', '.']
        expected_dr_3 = ['root', 'det', 'amod', 'nsubj', 'ccomp', 
                         'mark', 'nsubj', 'advcl', 'punct']
        expected_smwe_3 = ['FALSE', 'FALSE', 'FALSE', 'FALSE', 'FALSE', 
                           'FALSE', 'FALSE', 'FALSE', 'FALSE']
        expected_ss_3 = ['v.social', 'NONE', 'NONE', 'n.PERSON', 'v.cognition', 
                         'p.Time', 'NONE', 'v.possession', 'NONE']
        tags_3 = [expected_upos_3, expected_xpos_3, expected_dr_3, 
                  expected_smwe_3, expected_ss_3]

        for tag_index, tag_name in enumerate(tag_2_index):
            streusle_reader = StreusleDatasetReader(lazy=lazy, 
                                                    tag_name=tag_name)
            instances = streusle_reader.read(str(self.DATA_DIR / "example_data.conll"))
            instances = ensure_list(instances)
            assert len(instances) == 3

            fields = instances[0].fields
            tokens = [t.text for t in fields["tokens"].tokens]
            assert tokens == expected_tokens_1
            assert fields["tags"].labels == tags_1[tag_index], tag_name

            fields = instances[1].fields
            tokens = [t.text for t in fields["tokens"].tokens]
            assert tokens == expected_tokens_2
            assert fields["tags"].labels == tags_2[tag_index], tag_name
            
            fields = instances[2].fields
            tokens = [t.text for t in fields["tokens"].tokens]
            assert tokens == expected_tokens_3
            assert fields["tags"].labels == tags_3[tag_index], tag_name
        