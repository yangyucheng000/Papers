import math
import numpy as np
from typing import List, Optional, Set
import mindspore
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, initializer,Normal
from mindspore.ops import operations as P
from mindspore import Parameter,Tensor,ops,dtype as mstype
from mindspore.common.initializer import HeUniform
from mindspore.common.initializer import Uniform
import inspect
from collections import OrderedDict
from typing import Callable,Dict, Any
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        minval = Tensor(-bound, mindspore.float32)
        maxval = Tensor(bound, mindspore.float32)
        tensor.set_data(ops.uniform(tensor.shape, minval, maxval, seed=5))
def print_elements(var):
    if isinstance(var, (list, tuple)):
        for elem in var:
            print(elem)
    elif isinstance(var, dict):
        for key, value in var.items():
            print(f'{key}: {value}')
    elif hasattr(var, '__dict__'):
        for attr, value in vars(var).items():
            print(attr, type(value))
            # print(f'{attr}: {value}')
    else:
        print(var)
class RGCN(nn.Cell):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = Tensor(initializer(XavierUniform(), [num_relations, 100]),mindspore.float32)
        print("before RGCNConv")
        self.conv1 = RGCNConv(100, 100, num_relations * 2, num_bases)
        print("after RGCNConv")
        self.conv2 = RGCNConv(100, 100, num_relations * 2, num_bases)
        self.dropout_ratio = dropout
        self.relu=P.ReLU()
        self.dropout = nn.Dropout(keep_prob=1 - dropout)


    def construct(self, entity, edge_index, edge_type, edge_norm):
        print(" in  RGCN construct")
        x = self.entity_embedding(entity)
        x = self.conv1(x, edge_index, edge_type, edge_norm)
        x = self.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        return x

    def distmult(self, embedding, triplets):
        print(" in  dismult")
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = P.ReduceSum()(s * r * o, 1)
        return score

    def score_loss(self, embedding, triplets, target):
        print(" in  score_loss")
        score = self.distmult(embedding, triplets)
        return score, P.SigmoidCrossEntropyWithLogits()(score, target)

    def reg_loss(self, embedding):
        return P.ReduceMean()(embedding**2) + P.ReduceMean()(self.relation_embedding**2)
class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class: Any = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    def inspect(self, func: Callable, pop_first: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params
        return params

    def keys(self, func_names=None):
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def __implements__(self, cls, func_name):
        if cls.__name__ == 'MessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name):
        return self.__implements__(self.base_class.__class__, func_name)
    def distribute(self, func_name: str, kwargs: Dict[str, Any]):
        out = {}
        for key, param in self.params[func_name].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out
class MessagePassing(nn.Cell):
    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }
    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):
        super(MessagePassing, self).__init__()
        print("after init")
        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

        # record the class implements which methods and their args
        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate', 'update']).difference(self.special_args)

        # Support for "fused" message passing.
        # test if the class implements the message_and_aggregate method
        self.fuse = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

    def message(self, x_j: Tensor) -> Tensor:

        return x_j
    def message_and_aggregate(self, adj_t: mindspore.COOTensor) -> Tensor:
        raise NotImplementedError

    # def expand_left(tensor, dim, dims):
    #     shape = list(tensor.shape)
    #     shape.insert(dim, 1)
    #     tensor = ops.Reshape()(tensor, shape)
    #     multiples = [1] * dims
    #     multiples[dim] = -1
    #     return ops.Tile()(tensor, multiples)
    #
    # def segment_csr(src, indptr, out=None, reduce='sum'):
    #     if reduce != 'sum':
    #         raise ValueError("Only 'sum' reduction is supported")
    #     if out is None:
    #         max_index = ops.ReduceMax()(indptr)
    #         out = ops.ZerosLike()(src)
    #         out = ops.Reshape()(out, (max_index, -1))
    #     for i in range(1, len(indptr)):
    #         segment = src[indptr[i - 1]:indptr[i]]
    #         out[i - 1] = ops.ReduceSum()(segment, 0)
    #     return out
    def tensor_scatter(inputs, index, dim, dim_size=None, reduce=None):
        # 创建一个全零的目标张量
        output = ops.zeros(dim_size, inputs.dtype)

        # 使用scatter操作将inputs中的元素根据index的索引值散布到output中
        output = ops.scatter(output, dim, index, inputs)
        print("in tensor_scatter")
        # 如果reduce参数指定了聚合操作，则对output进行相应的聚合操作
        if reduce == 'sum':
            output = ops.reduce_sum(output, dim)
        elif reduce == 'mean':
            output = ops.reduce_mean(output, dim)
        elif reduce == 'max':
            output = ops.reduce_max(output, dim)
        elif reduce == 'min':
            output = ops.reduce_min(output, dim)

        return output
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        # if dim_size is None:
        #     dim_size = inputs.shape[self.node_dim]
        # index = index.reshape((-1, 1))
        # updates = inputs.reshape((-1, inputs.shape[-1]))
        # output_shape = (dim_size, inputs.shape[-1])
        mindspore.ops.Print(inputs,"in aggregate")
        return self.tensor_scatter(inputs, index, self.node_dim, dim_size,
                       self.aggr)


    def update(self, inputs: Tensor) -> Tensor:
        return inputs
    def __check_input__(self, edge_index, size):
        the_size = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == mstype.int64
            assert edge_index.dim() == 2
            assert edge_index.shape[0] == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size
        elif isinstance(edge_index, mindspore.COOTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `mindspore.COOTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed COOTensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.shape[1]
            the_size[1] = edge_index.shape[0]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.shape[self.node_dim]
        elif the_size != src.shape[self.node_dim]:
            raise ValueError(
                (f'Encountered tensor with size {src.shape[self.node_dim]} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, mindspore.Tensor):
            index = edge_index[dim]
            return src.index_select(dim, index)
        # if isinstance(edge_index, mindspore.Tensor):
        #     index = edge_index[dim]
        #     return src.gather(self.node_dim, index)
        # elif isinstance(edge_index, mindspore.COOTensor):
        #     if dim == 1:
        #         rowptr = edge_index.storage.rowptr()
        #         rowptr = mindspore.ops.ExpandDims()(rowptr, self.node_dim)
        #         return mindspore.ops.GatherV2()(src, rowptr, self.node_dim)
        #     elif dim == 0:
        #         col = edge_index.storage.col()
        #         return src.gather(self.node_dim, col)
        raise ValueError
    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, None)
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], None)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]
    #代码运行这
                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        else:
            # MindSpore doesn't have a direct equivalent for PyTorch's SparseTensor
            # You might need to implement it yourself or find a workaround
            pass

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out
    def propagate(self, edge_index, size=None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, mindspore.COOTensor) and self.fuse
                and not self.__explain__):  # MindSpore currently does not support SparseTensor
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)
            print(" in if")
            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        #代码运行elif
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)
            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = mindspore.ops.Concat(0)((edge_mask, loop))
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)

            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)





class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.att_l = Parameter(Tensor(out_channels, mindspore.float32))
        self.att_l = Parameter(Tensor(np.random.rand(out_channels)), name="att_l")

        self.att_r = Parameter(Tensor(np.random.rand(out_channels)), name="att_r")
        self.lin_l = nn.Dense(in_channels, out_channels)

        self.num_relations = num_relations
        self.num_bases = num_bases
        self.basis = Parameter(Tensor(np.random.rand(num_bases, in_channels, out_channels)),name="basis")
        self.att = Parameter(Tensor(np.random.rand(num_relations, num_bases)), name="att")

        if root_weight:
            # self.root = Parameter(initializer(XavierUniform(), [in_channels, out_channels]))
            self.root = Parameter(Tensor(np.random.rand(in_channels, out_channels)), name="root")
        else:
            self.register_parameter('root', None)

        if bias:
            # self.bias = Parameter(initializer(Normal(0,0.02), [out_channels]))
            self.bias =Parameter(Tensor(np.random.rand(out_channels)), name="bias")
            # self.bias = Parameter(Tensor(out_channels, mindspore.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def construct(self, x, edge_index, edge_type, edge_norm=None, size=None):


        result= self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)
        return result


    def reset_parameters(self):

        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)



    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
