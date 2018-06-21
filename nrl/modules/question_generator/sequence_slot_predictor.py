from nrl.modules.question_generator.question_generator import QuestionGenerator

from typing import List

import torch

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Linear, Dropout, Embedding, LSTMCell
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.modules import TimeDistributed

from nrl.util.model_utils import block_orthonormal_initialization

@QuestionGenerator.register("sequence")
class SequenceSlotPredictor(QuestionGenerator):
    def __init__(self,
            vocab: Vocabulary,
            slot_labels: List[str],
            input_dim: int,
            dim_slot_hidden: int = 100,
            dim_rnn_hidden: int = 200,
            dim_embedding: int = 100,
            rnn_layers: int = 1,
            recurrent_dropout: float = 0.1,
            highway: bool = True,
            share_rnn_cell: bool =  True,
            share_slot_hidden: bool = False):
        super(SequenceSlotPredictor, self).__init__(vocab, slot_labels, input_dim)
        self._dim_embedding = dim_embedding
        self._dim_slot_hidden = dim_slot_hidden
        self._dim_rnn_hidden = dim_rnn_hidden
        self._rnn_layers = rnn_layers
        self._recurrent_dropout = recurrent_dropout


        slot_embedders = []
        for i, n in enumerate(self._slot_labels[:-1]):
            num_labels = self._vocab.get_vocab_size("slot_%s"%n)
            assert num_labels > 0, "Slot named %s has 0 vocab size"%(n)
            embedder = Embedding(num_labels, self._dim_embedding)
            self.add_module('embedder_%s'%n, embedder)
            slot_embedders.append(embedder)

        self._slot_embedders = slot_embedders

        self._highway = highway

        rnn_cells = []
        highway_nonlin = []
        highway_lin = []
        for l in range(self._rnn_layers):
            layer_cells = []
            layer_highway_nonlin = []
            layer_highway_lin = []
            shared_cell = None
            layer_input_size = 2 * self._input_dim + self._dim_embedding if l == 0 else self._dim_rnn_hidden
            for i, n in enumerate(self._slot_labels):
                if share_rnn_cell:
                    if shared_cell is None:
                        shared_cell = LSTMCell(layer_input_size, self._dim_rnn_hidden)
                        self.add_module('layer_%d_cell'%l, shared_cell)
                        if highway:
                            shared_highway_nonlin = Linear(layer_input_size + self._dim_rnn_hidden, self._dim_rnn_hidden)
                            shared_highway_lin = Linear(layer_input_size, self._dim_rnn_hidden, bias = False)
                            self.add_module('layer_%d_highway_nonlin'%l, shared_highway_nonlin)
                            self.add_module('layer_%d_highway_lin'%l, shared_highway_lin)
                    layer_cells.append(shared_cell)
                    if highway:
                        layer_highway_nonlin.append(shared_highway_nonlin)
                        layer_highway_lin.append(shared_highway_lin)
                else:
                    cell = LSTMCell(layer_input_size, self._dim_rnn_hidden)
                    cell.weight_ih.data.copy_(block_orthonormal_initialization(layer_input_size, self._dim_rnn_hidden, 4).t())
                    cell.weight_hh.data.copy_(block_orthonormal_initialization(self._dim_rnn_hidden, self._dim_rnn_hidden, 4).t())
                    self.add_module('layer_%d_cell_%s'%(l, n), cell)
                    layer_cells.append(cell)
                    if highway:
                        nonlin = Linear(layer_input_size + self._dim_rnn_hidden, self._dim_rnn_hidden)
                        lin = Linear(layer_input_size, self._dim_rnn_hidden, bias = False)
                        nonlin.weight.data.copy_(block_orthonormal_initialization(layer_input_size + self._dim_rnn_hidden, self._dim_rnn_hidden, 1).t())
                        lin.weight.data.copy_(block_orthonormal_initialization(layer_input_size, self._dim_rnn_hidden, 1).t())
                        self.add_module('layer_%d_highway_nonlin_%s'%(l, n), nonlin)
                        self.add_module('layer_%d_highway_lin_%s'%(l, n), lin)
                        layer_highway_nonlin.append(nonlin)
                        layer_highway_lin.append(lin)
                        
            rnn_cells.append(layer_cells)
            highway_nonlin.append(layer_highway_nonlin)
            highway_lin.append(layer_highway_lin)

        self._rnn_cells = rnn_cells
        if highway:
            self._highway_nonlin = highway_nonlin
            self._highway_lin = highway_lin

        shared_slot_hidden = None
        slot_hiddens = []
        slot_preds = []
        slot_num_labels = []
        for i, n in enumerate(self._slot_labels):
            num_labels = self._vocab.get_vocab_size("slot_%s"%n)
            slot_num_labels.append(num_labels)

            if share_slot_hidden:
                if shared_slot_hidden is None:
                    shared_slot_hidden = Linear(self._dim_rnn_hidden, self._dim_slot_hidden)
                    self.add_module('slot_hidden', shared_slot_hidden)
                slot_hiddens.append(shared_slot_hidden)
            else:
                slot_hidden = Linear(self._dim_rnn_hidden, self._dim_slot_hidden)
                slot_hiddens.append(slot_hidden)
                self.add_module('slot_hidden_%s'%n, slot_hidden)

            slot_pred = Linear(self._dim_slot_hidden, num_labels)
            slot_preds.append(slot_pred)
            self.add_module('slot_pred_%s'%n, slot_pred)

        self._slot_hiddens = slot_hiddens
        self._slot_preds = slot_preds
        self._slot_num_labels = slot_num_labels

        self._start_symbol = Parameter(torch.Tensor(self._dim_embedding).normal_(0, 1))

    def forward(self, span_reps, slot_labels = None, **kwargs):
        batch_size, max_spans, repsize = span_reps.size()
        span_reps = span_reps.view(-1, repsize)

        curr_embedding = self._start_symbol.view(1, -1).expand(span_reps.size(0), -1)
        curr_mem = []
        for l in range(self._rnn_layers):
            curr_mem.append((Variable(span_reps.data.new().resize_(span_reps.size(0), self._dim_rnn_hidden).zero_()),
                             Variable(span_reps.data.new().resize_(span_reps.size(0), self._dim_rnn_hidden).zero_())))

        slot_logits = []
        for i, n in enumerate(self._slot_labels):
            next_mem  = []
            curr_input = torch.cat([span_reps, curr_embedding], -1)
            for l in range(self._rnn_layers):
                new_h, new_c = self._rnn_cells[l][i](curr_input, curr_mem[l])
                if self._recurrent_dropout > 0:
                    new_h = F.dropout(new_h, p = self._recurrent_dropout, training = self.training)
                next_mem.append((new_h, new_c))
                if self._highway:
                    nonlin = self._highway_nonlin[l][i](torch.cat([curr_input, new_h], -1))
                    gate = F.sigmoid(nonlin)
                    curr_input = gate * new_h + (1. - gate) * self._highway_lin[l][i](curr_input)
                else:
                    curr_input = new_h
            curr_mem = next_mem
            hidden = F.relu(self._slot_hiddens[i](new_h))
            logits = self._slot_preds[i](hidden)
            slot_logits.append(logits.view(batch_size, max_spans, -1))

            if i < len(self._slot_labels) - 1:
                if self.training:
                    curr_embedding = self._slot_embedders[i](slot_labels[i]).view(span_reps.size(0), -1)
                else:
                    _, max_inds = logits.max(-1)
                    curr_embedding = self._slot_embedders[i](max_inds)

        return slot_logits

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SequenceSlotPredictor':
        slot_labels = params.pop("slot_labels")
        input_dim = params.pop("input_dim")
        dim_slot_hidden = params.pop("dim_slot_hidden")
        share_slot_hidden = params.pop("share_slot_hidden", False)
        rnn_layers = params.pop("rnn_layers", 1)
        share_rnn_cell = params.pop("share_rnn_cell", True)
        dim_rnn_hidden = params.pop("dim_rnn_hidden", 200)
        dim_slot_hidden = params.pop("dim_slot_hidden", 100)
        dim_embedding = params.pop("dim_embedding", 100)
        recurrent_dropout = params.pop("recurrent_dropout", 0.1)

        params.assert_empty(cls.__name__)

        return SequenceSlotPredictor(vocab, slot_labels, input_dim=input_dim, share_slot_hidden=share_slot_hidden, rnn_layers = rnn_layers, share_rnn_cell = share_rnn_cell, dim_rnn_hidden = dim_rnn_hidden, dim_slot_hidden = dim_slot_hidden, dim_embedding = dim_embedding, recurrent_dropout = recurrent_dropout)
