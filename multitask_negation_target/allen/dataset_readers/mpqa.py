import logging
from typing import Dict, Optional, List, Iterable
import re
import itertools

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from overrides import overrides

logger = logging.getLogger(__name__)

def _is_divider(line: str) -> bool:
    if line.strip() == '':
        return True
    elif re.search(r'^## Doc ID', line):
        return True
    elif re.search(r'^## Sent ID', line):
        return True
    else:
        return False

@DatasetReader.register("mpqa")
class MPQADatasetReader(DatasetReader):
    '''
    The CONLL file should have the following structure:
    `TOKEN#GOLD LABEL`
    Where each text is sperated by a blank new line and that each text has an 
    associated `## Sent ID` line at the start of the text and potential a 
    `## Doc ID` line. An example of the file is below:
    `
    ## Doc ID: 20020304/20.42.01-25605
    ## Sent ID: 20020304/20.42.01-25605:1
    The O
    laptop B-0
    case L-0
    was O
    great O
    and O
    cover O
    was O
    rubbish O
    ## Sent ID: 20020304/20.42.01-25605:2
    The O
    laptop B-0
    case L-0
    was O
    great O
    and O
    cover U-1
    was O
    rubbish O
    `
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the sequence labels.
    '''
    def __init__(self, lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conll_file:
            logger.info("Reading MPQA instances from CONLL "
                        "dataset at: %s", file_path)
            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(conll_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if is_divider:
                    continue
                fields = [line.strip().split('\t') for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                fields = [list(field) for field in zip(*fields)]
                tokens_ = fields[0]
                tags = fields[1]

                # TextField requires ``Token`` objects
                tokens = [Token(token) for token in tokens_]

                yield self.text_to_instance(tokens, tags)
    
    def text_to_instance(self, tokens: List[Token],
                         tags: Optional[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer 
        in this class.
        """
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        # Metadata field
        metadata_dict = {"words": [x.text for x in tokens]}
        instance_fields["metadata"] = MetadataField(metadata_dict)

        if tags is not None:
            instance_fields['tags'] = SequenceLabelField(tags, sequence, 
                                                         self.label_namespace)

        return Instance(instance_fields)