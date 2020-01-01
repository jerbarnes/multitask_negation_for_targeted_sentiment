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

def skip_tokens(token: str) -> bool:
    '''
    Used to skip tokens that are actually lines like:
    `# domain - BOOKS\n` or
    `# sentence - I boug....\n`
    '''
    if (re.search(r'^#token.*', token) or 
        re.search(r'^# document - .*', token) or 
        re.search(r'^# domain - .*', token) or
        re.search(r'^# .*', token)):
        return True
    return False

@DatasetReader.register("negation_speculation")
class NegationSpeculationDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD NEGATION-TAG SPECULATION-TAG

    Where the SPECULATION-TAG is optional.
    With a blank line indicating the end of each sentence
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.

    The NEGATION-TAG and SPECULATION-TAG are expected to be in BIO coding scheme 
    where B is a token starting a span, I is a token continuing a span, and
    O is a token outside of a span.

    Custom version of:
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/conll2000.py
    
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``negation``)
        Specify `negation`, or `speculation` to have that tag loaded into the instance field `tag`.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """

    _VALID_LABELS = {"negation", "speculation"}

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "negation", lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError(f"unknown tag label type: {tag_label}")
        self.tag_label = tag_label
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
                    lines = [line for line in lines if not skip_tokens(line)]
                    # This happens when there are no tokens for an empty sentence
                    if not lines:
                        continue
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    if self.tag_label == 'speculation' or len(fields) == 3:
                        tokens_, negation_tags, speculation_tags = fields
                    else:
                        tokens_, negation_tags = fields[0], fields[1]
                        speculation_tags = None

                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, negation_tags, speculation_tags)

    def text_to_instance(self, tokens: List[Token], 
                         negation_tags: List[str] = None, 
                         speculation_tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        # Add "tag label" to instance
        if self.tag_label == "negation":
            instance_fields["tags"] = SequenceLabelField(negation_tags, sequence, 
                                                         self.label_namespace)
        elif self.tag_label == "speculation":
            instance_fields["tags"] = SequenceLabelField(speculation_tags, sequence, 
                                                         self.label_namespace)
        return Instance(instance_fields)