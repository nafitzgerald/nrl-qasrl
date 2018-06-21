from nrl.modules.question_generator.question_generator import QuestionGenerator

from typing import List

from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.modules import TimeDistributed

@QuestionGenerator.register("local")
class LocalSlotPredictor(QuestionGenerator):
    def __init__(self,
            vocab: Vocabulary,
            slot_labels : List[str],
            input_dim : int,
            dim_slot_hidden: int = 100,
            share_slot_hidden: bool = False):
        super(LocalSlotPredictor, self).__init__(vocab, slot_labels, input_dim)
        self._dim_slot_hidden = dim_slot_hidden

        shared_slot_hidden = None
        slot_hiddens = []
        slot_preds = []
        slot_num_labels = []
        for i, n in enumerate(self._slot_labels):
            num_labels = self._vocab.get_vocab_size("slot_%s"%n)
            assert num_labels > 0, "Slot named %s has 0 vocab size"%(n)
            slot_num_labels.append(num_labels)

            if share_slot_hidden:
                if shared_slot_hidden is None:
                    shared_slot_hidden = TimeDistributed(Linear(2*self._input_dim, self._dim_slot_hidden))
                    self.add_module('slot_hidden', shared_slot_hidden)
                slot_hiddens.append(shared_slot_hidden)
            else:
                slot_hidden = TimeDistributed(Linear(2*self._input_dim, self._dim_slot_hidden))
                slot_hiddens.append(slot_hidden)
                self.add_module('slot_hidden_%s'%n, slot_hidden)

            slot_pred = TimeDistributed(Linear(self._dim_slot_hidden, num_labels))
            slot_preds.append(slot_pred)
            self.add_module('slot_pred_%s'%n, slot_pred)

        self._slot_hiddens = slot_hiddens
        self._slot_preds = slot_preds
        self._slot_num_labels = slot_num_labels

    def forward(self, span_reps, **kwargs):
        slot_logits = []
        for i, n in enumerate(self._slot_labels):
            hidden = F.relu(self._slot_hiddens[i](span_reps))
            logits = self._slot_preds[i](hidden)
            slot_logits.append(logits)
        return slot_logits


    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'LocalSlotPredictor':
        slot_labels = params.pop("slot_labels")
        input_dim = params.pop("input_dim")
        dim_slot_hidden = params.pop("dim_slot_hidden")
        share_slot_hidden = params.pop("share_slot_hidden", False)

        return LocalSlotPredictor(vocab, slot_labels, input_dim=input_dim, dim_slot_hidden=dim_slot_hidden, share_slot_hidden=share_slot_hidden)
