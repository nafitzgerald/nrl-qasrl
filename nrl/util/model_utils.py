import torch, numpy

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

def orthonormal_initialization(dim_in, dim_out, factor=1.0, seed=None, dtype='float64'):
    rng = numpy.random.RandomState(seed)
    if dim_in == dim_out:
        M = rng.randn(*[dim_in, dim_out]).astype(dtype)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        param = torch.Tensor(Q * factor)
    else:
        M1 = rng.randn(dim_in, dim_in).astype(dtype)
        M2 = rng.randn(dim_out, dim_out).astype(dtype)
        Q1, R1 = numpy.linalg.qr(M1)
        Q2, R2 = numpy.linalg.qr(M2)
        Q1 = Q1 * numpy.sign(numpy.diag(R1))
        Q2 = Q2 * numpy.sign(numpy.diag(R2))
        n_min = min(dim_in, dim_out)
        param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * factor
        param = torch.Tensor(param)
    return param

def block_orthonormal_initialization(dim_in, dim_out, num_blocks, factor=1.0, seed=None, dtype='float64'):
    param = torch.cat([orthonormal_initialization(dim_in, dim_out) for i in range(num_blocks)], 1)
    return param

