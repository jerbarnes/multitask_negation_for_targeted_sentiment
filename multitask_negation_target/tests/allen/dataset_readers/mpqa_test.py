from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
import pytest

from multitask_negation_target.allen.dataset_readers.mpqa import MPQADatasetReader
from multitask_negation_target.tests import util

class TestMPQADatasetReader():
    DATA_DIR = util.FIXTURES_ROOT / "allen" / "dataset_readers" / "mpqa"

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool):
        without_doc_sent__and_space_ids = str(self.DATA_DIR / "mpqa_without_doc_sent_alt.conll")
        without_doc_sent_ids = str(self.DATA_DIR / "mpqa_without_doc_sent.conll")
        with_doc_sent_ids = str(self.DATA_DIR / "mpqa.conll")
        test_fps = [with_doc_sent_ids, without_doc_sent_ids, without_doc_sent__and_space_ids]
        for test_fp in test_fps:
            reader = MPQADatasetReader(lazy=lazy)
            instances = reader.read(test_fp)
            instances = ensure_list(instances)
            assert len(instances) == 3


            instance1 = {"tokens": ["[", "Independence", "of", "]"],
                        "tags": ["O", "B-positive", "L-positive", "O"]}
            instance2 = {"tokens": ["The", "Iran", "."],
                        "tags": ["O", "U-neutral", "O"]}
            instance3 = {"tokens": ["towards", "the", "Islamic", "Republic", "."],
                        "tags": ["O", "B-negative", "I-negative", "L-negative", "O"]}

            assert len(instances) == 3
            fields = instances[0].fields
            assert [t.text for t in fields["tokens"]] == instance1["tokens"]
            assert fields['tags'].labels == instance1["tags"]
            assert fields['metadata']['words'] == instance1["tokens"]
            
            fields = instances[1].fields
            assert [t.text for t in fields["tokens"]] == instance2["tokens"]
            assert fields['tags'].labels == instance2["tags"]
            assert fields['metadata']['words'] == instance2["tokens"]

            fields = instances[2].fields
            assert [t.text for t in fields["tokens"]] == instance3["tokens"]
            assert fields['tags'].labels == instance3["tags"]
            assert fields['metadata']['words'] == instance3["tokens"]