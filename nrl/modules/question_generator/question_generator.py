import torch

from typing import List

from allennlp.data import Vocabulary
from allennlp.common import Params, Registrable

class QuestionGenerator(torch.nn.Module, Registrable):
    def __init__(self,
            vocab: Vocabulary,
            slot_labels : List[str],
            input_dim: int):
        super(QuestionGenerator, self).__init__()
        self._vocab = vocab
        self._slot_labels = slot_labels
        self._input_dim = input_dim

    def get_slot_labels(self):
        return self._slot_labels

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'QuestionGenerator':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(vocab, params)
