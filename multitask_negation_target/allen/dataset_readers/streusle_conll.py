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
    if line.strip() == '':
        return True
    elif re.search(r'^#', line):
        return True
    else:
        return False

@DatasetReader.register("streusle")
class StreusleDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line has the following 
    format from `Streusle <https://github.com/nert-nlp/streusle/blob/master/CONLLULEX.md>`_

    From that format there is only 5 columns we are intrested in:
    1. UPOS - Which is column number 4
    2. XPOS - Which is column number 5
    3. Dependency relations (DR) - Which is column number 8
    4. Strong Multi Word Expresion (SMWE) - Which is column number 11
    5. Super Sense tagging (SS) - Which is column number 14

    Each line/sentence is seperated by a blank line where the top few lines 
    contains meta data in the form of `# some metadata`. 

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the tag.
    tag_name : ``str`` the name of the tag you want the dataset reader to 
        produce e.g. `UPOS` for universal POS tags. List of acceptable tags are:
        ``['UPOS', 'XPOS', 'DR', 'SMWE', 'SS']``
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 tag_name: str = "UPOS", lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

        self.tag_name = tag_name
        tag_2_column = {'UPOS': 3, 'XPOS': 4, 'DR': 7, 'SMWE': 10, 'SS': 13}
        self.tag_column = tag_2_column[tag_name]

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from Streusle file at: %s", file_path)
            logger.info("Name of tags is %s", self.tag_name)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if is_divider:
                    continue

                fields = [line.strip().split('\t') for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                fields = [list(field) for field in zip(*fields)]
                tokens_ = fields[1]
                tags = fields[self.tag_column]
                if self.tag_name == 'SMWE':
                    temp_tags = []
                    for tag in tags:
                        if tag == '_':
                            temp_tags.append('FALSE')
                        else:
                            temp_tags.append('TRUE')
                    tags = temp_tags
                elif self.tag_name == 'SS':
                    temp_tags = []
                    for tag in tags:
                        if tag == '_':
                            temp_tags.append('NONE')
                        else:
                            temp_tags.append(tag)
                    tags = temp_tags
                elif self.tag_name == 'DR':
                    temp_tags = []
                    for tag in tags:
                        if tag == '_':
                            temp_tags.append('NONE')
                        else:
                            temp_tags.append(tag)
                    tags = temp_tags
                
                # TextField requires ``Token`` objects
                tokens = [Token(token) for token in tokens_]

                yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None
                         ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            instance_fields["tags"] = SequenceLabelField(tags, sequence, 
                                                         self.label_namespace)
        return Instance(instance_fields)