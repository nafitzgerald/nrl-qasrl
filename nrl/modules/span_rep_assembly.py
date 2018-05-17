import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.modules import TimeDistributed

class SpanRepAssembly(torch.nn.Module):
    def __init__(self,
            embA_size: int,
            embB_size: int,
            hidden_dim: int):
        super(SpanRepAssembly, self).__init__()

        self.embA_size = embA_size
        self.embB_size = embB_size
        self.hidden_dim = hidden_dim

        self.hiddenA = TimeDistributed(Linear(embA_size, hidden_dim))
        self.hiddenB = TimeDistributed(Linear(embB_size, hidden_dim, bias=False))

    def forward(self,
            embA: torch.Tensor,
            embB: torch.Tensor,
            maskA: torch.Tensor,
            maskB: torch.Tensor):

        hiddenA = self.hiddenA(embA) # B x Ta X H
        hiddenB = self.hiddenB(embB) # B x Tb X H

        combined, mask = cross_product_combine(hiddenA, hiddenB, maskA, maskB, ordered = True)

        return combined, mask

def cross_product_combine(hiddenA, hiddenB, maskA, maskB, ordered = False):
        batchA, timeA, embsizeA = hiddenA.size()
        batchB, timeB, embsizeB = hiddenB.size()

        assert batchA == batchB
        assert embsizeA == embsizeB

        if ordered:
            assert timeA == timeB
            out_num = int((timeA * (timeA+1)) / 2)

            hiddenA_data = hiddenA.data if isinstance(hiddenA, Variable) else hiddenA
            indexA = hiddenA_data.new().long().resize_(out_num).copy_(torch.Tensor([start for start in range(timeA) for i in range(start, timeA)]))
            indexA = Variable(indexA) if isinstance(hiddenA, Variable) else indexA
            hiddenA_rep = hiddenA.index_select(1, indexA)
            maskA_rep = maskA.index_select(1, indexA)

            indexB = hiddenA_data.new().long().resize_(out_num).copy_(torch.Tensor([i for start in range(timeA) for i in range(start, timeA)]))
            indexB = Variable(indexB) if isinstance(hiddenB, Variable) else indexB
            hiddenB_rep = hiddenB.index_select(1, indexB)
            maskB_rep = maskB.index_select(1, indexB)
        else:
            hiddenA_rep = hiddenA.view(batchA, timeA, 1, embsizeA)
            hiddenB_rep = hiddenB.view(batchB, 1, timeB, embsizeB)
            maskA_rep = maskA.view(batchA, timeA, 1)
            maskB_rep = maskB.view(batchB, 1, timeB)

        combined = (hiddenA_rep + hiddenB_rep).view(batchA, -1, embsizeA)
        mask = (maskA_rep * maskB_rep).view(batchA, -1)

        return combined, mask



 
