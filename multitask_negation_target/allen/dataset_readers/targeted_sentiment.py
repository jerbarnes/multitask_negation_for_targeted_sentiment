from typing import Dict, List, Sequence, Iterable
import itertools
import logging
import re

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    return line.strip() == ""

@DatasetReader.register("targeted_sentiment")
class TargetedSentimentDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following 
    format: WORD SENTIMENT-TAG

    With a blank line indicating the end of each sentence
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The SENTIMENT-TAG values will get loaded into the ``"tags"`` 
    ``SequenceLabelField``.

    The SENTIMENT-TAG can be in either BIO or BIOUL coding scheme.

    Custom version of:
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/conll2000.py
    
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the SENTIMENT-TAG.
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "negation", lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, sentiment_tags = fields

                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, sentiment_tags)

    def text_to_instance(self, tokens: List[Token], 
                         sentiment_tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})


        instance_fields["tags"] = SequenceLabelField(sentiment_tags, sequence, 
                                                     self.label_namespace)
        return Instance(instance_fields)