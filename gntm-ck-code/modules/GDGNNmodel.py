# import torch.nn as nn
import mindspore
import mindspore as ms
import numpy as np

from .encoder import *
# from torch_scatter import scatter
# from torch_sparse import SparseTensor
from mindspore import  SparseTensor,nn,Parameter,Tensor,ParameterTuple
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
def Expectation_log_Dirichlet(gamma):
    #E_{Dir(\theta| \gamma)}[\log \theta] = \Psi(\gamma) - \Psi(|\gamma|)
    return torch.digamma(gamma) - torch.digamma(gamma.sum(dim=-1, keepdim=True))
import math

def print_elements(var):
    if isinstance(var, (list, tuple)):
        for elem in var:
            print(elem)
    elif isinstance(var, dict):
        for key, value in var.items():
            print(f'{key}: {value}')
    elif hasattr(var, '__dict__'):
        for attr, value in vars(var).items():
            # print(attr, type(value))
            print(f'{attr}: {value}')
    else:
        print(var)
def scatter_sum(src, index, dim_size):
    scatter = P.ScatterAdd()
    output = mindspore.ops.zeros(dim_size, mindspore.float32)
    return scatter(output, index, src)
class GDGNNModel(nn.Cell):
    def __init__(self, args, word_vec, whole_edge):
        super(GDGNNModel, self).__init__()
        # encoder
        self.args = args
        # 用来设置节点的特征矩阵
        if word_vec is None:
            print("in if before parameter")
            self.word_vec = Parameter(Tensor(np.zeros(args.vocab_size, args.ni),dtype=mindspore.float32))
            print("in if after parameter")
            self.word_vec.set_data(Normal(0.01)(self.word_vec.shape()))
        else:
            print("in else before parameter")
            self.word_vec = Parameter(Tensor(word_vec))
            print("in else after parameter")
            if args.fixing:
                self.word_vec.requires_grad = False

        if args.prior_type == 'Dir2':   
            self.encoder = RGCNDir2encoder(args, self.word_vec)
        else:
            raise ValueError("the specific prior type is not supported")

        # decoder
        self.word_vec_beta = Parameter(Tensor(np.random.normal(0, 0.01, [args.vocab_size, args.ni]), mindspore.float32))
        self.topic_vec = Parameter(Tensor(np.random.normal(0, 1, [args.num_topic, args.ni]), mindspore.float32))
        self.topic_edge_vec = Parameter(
            Tensor(np.random.normal(0, 1, [args.num_topic, 2 * args.ni]), mindspore.float32))
        self.noedge_vec = Parameter(Tensor(np.random.normal(0, 0.01, [1, 2 * args.ni]), mindspore.float32))

        # MindSpore不支持torch.tensor，你可以使用Tensor来创建张量
        self.maskrate = Tensor(np.ones([1]) / args.num_topic, mindspore.float32)

        self.topic_linear = nn.Dense(3 * args.ni, 64, has_bias=False)

        # MindSpore不支持.parameters()函数，但你可以直接访问参数
        self.enc_params = list(self.encoder.get_parameters())
        self.dec_params = [self.word_vec_beta, self.topic_vec, self.topic_edge_vec, self.noedge_vec] + list(
            self.topic_linear.get_parameters())
        self.reset_parameters()


    def construct(self, batch_data, batch_gcn_inputs, batch_gcn_idxs, batch_idx2sent, doc_g, batch_doc_ids):
        # print("construct  run")
        # print_elements(batch_gcn_inputs[0])
        # exit(0)
        # //接受到相同的数据
        idx_x = batch_data.x
        x_batch = batch_data.x_batch
        idx_w = batch_data.x_w
        edge_w = batch_data.edge_w
        edge_id = batch_data.edge_id
        edge_id_batch =batch_data.edge_id_batch
        edge_index = batch_data.edge_index
        size = int(x_batch.max().item() + 1)
        # param, phi, gcn_recon, gcn_init = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w, batch_gcn_inputs,
        #                                                batch_gcn_idxs, batch_idx2sent)
        # 创建一个空列表来存储转换后的数据
        # ms_batch_gcn_inputs = []
        # for data in batch_gcn_inputs:
        #     ms_batch_gcn_inputs.append(list(data.values()))
        # 将每个Data对象的属性转换为Tensor


        # edge_index_list=[]
        # entity_list=[]
        # edge_type_list=[]
        # edge_norm_list=[]
        # for data in batch_gcn_inputs:
        #
        #     edge_index_list.append(data.edge_index)
        #     entity_list.append(data.entity)
        #     edge_type_list.append(data.edge_type)
        #     edge_norm_list.append(data.edge_norm)
        #
        #
        # param, phi, gcn_recon, gcn_init = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w, entity_list,edge_index_list,edge_type_list,edge_norm_list, batch_gcn_idxs, batch_idx2sent)
        # #####KL theta
        print("before encoder")
        param, phi, gcn_recon, gcn_init = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w, batch_gcn_inputs, batch_gcn_idxs, batch_idx2sent)
        print("after encoder")
        KL1 = self.encoder.KL_loss(param)  # param = (mean, logvar)  -> L_d^3

        #####KL(z)
        # (batch, max_length)
        if self.args.prior_type in ['Dir2', 'Gaussian']:  # -> logistic normal, gaussian
            theta = self.encoder.reparameterize(param)
            KL2 = P.ReduceSum()(phi * (P.Log()(phi / (theta[x_batch] + 1e-10) + 1e-10)), -1)

        # if self.args.prior_type == 'Dir':
        #     gamma = param[0] # (B, K)
        #     Elogtheta = Expectation_log_Dirichlet(gamma) # (B, K)
        #     KL2 = torch.sum(phi * ((phi + 1e-10).log() - Elogtheta[x_batch]), dim=-1)

        KL2 = myscattersum(idx_w * KL2, x_batch, size=size)  # B    -> L_d^4

        if not self.args.eval and self.args.prior_type in ['Dir2',
                                                           'Gaussian'] and self.args.iter_ < self.args.iter_threahold:
            phi = theta[x_batch]

        ##### generate structure  -> L_d^1
        W = self.get_W()  # (K, K)  after sigmod
        log_PW = (W).log()  # K,K    # logM
        log_NW = (1 - W).log()  # K,K     # log(1-M)
        p_phi = phi[edge_index, :]  # 2, B*len_edge, K    - phi -> (B*len, K)
        p_edge = P.ReduceSum()(P.MatMul()(p_phi[0], log_PW) * p_phi[1], -1)
        p_edge = myscattersum(p_edge, edge_id_batch, size=size)  # B

        neg_mask, neg_mask2 = adj_mask(x_batch, device=idx_w.device)

        neg_mask[edge_index[0], edge_index[1]] = 0

        n_edge = P.MatMul()(P.MatMul()(phi, log_NW), P.Transpose()(phi, (0, 1)))  # B*len, B*len
        n_edge1 = P.ReduceSum()(n_edge * neg_mask, -1)  # B*len
        n_edge1 = myscattersum(n_edge1, x_batch, size=size)  # B

        tmp = P.OnesLike()(edge_id_batch)
        NP = myscattersum(tmp, edge_id_batch, size=size)  # B

        NN = myscattersum(P.ReduceSum()(neg_mask, -1), x_batch, size=size)  # B

        recon_structure = - (p_edge + n_edge1 / (NN + 1e-6) * NP)

        # #### recon_word    -> L_d^2
        beta = self.get_beta()  # (K, V)

        q_z = RelaxedOneHotCategorical(temperature=self.args.temp, probs=phi)
        z = q_z.rsample([self.args.num_samp])  # (ns, B*len, K)

        z = hard_z(z, dim=-1)

        beta_s = beta[:, idx_x]  # K*(B*len) !TODO idx_x or idx_x
        beta_s = P.Transpose()(beta_s, (1, 0))  # (B*len)*K
        recon_word = P.ReduceSum()(phi * P.Log()(beta_s + 1e-6), -1)  # (B*len)
        recon_word = - myscattersum(idx_w * recon_word, x_batch)  # B

        ######recon_edge  -> L_d^2
        beta_edge = self.get_beta_edge(weight=False)

        edge_w = P.ExpandDims()(edge_w, 0)  # (1, B*len_edge)
        beta_edge_s = P.Transpose()(beta_edge[:, edge_id], (1, 0))  # (B*len_edge,K)

        z_edge_w = z[:, edge_index, :]  # (ns,2,B*len_edge, K)
        mask = z_edge_w > self.maskrate
        z_edge_w = z_edge_w * mask

        edge_w = P.ExpandDims()(edge_w, -1)  # (1, B*len_edge, 1)
        z_edge_w = edge_w * P.ReduceProd()(z_edge_w, 1)  # (ns,B*len_edge, K)
        beta_edge_s = P.ExpandDims()(beta_edge_s, 0) * z_edge_w  # (ns,B*len_edge,K)

        beta_edge_s = P.Transpose()(beta_edge_s, (1, 0, 2))  # (B*len_edge, ns,K)
        beta_edge_s = scatter(beta_edge_s, 0,edge_id_batch,beta_edge_s)  # (B, ns,K)
        beta_edge_s = P.Transpose()(beta_edge_s, (1, 0, 2))  # (ns,B,K)

        z_edge_w = P.Transpose()(z_edge_w, (1, 0, 2))  # (B*len,ns,K)
        z_edge_w = scatter(z_edge_w, 0,edge_id_batch,z_edge_w )  # (B,ns,K)
        z_edge_w = P.Transpose()(z_edge_w, (1, 0, 2))  # (ns,B,K)
        recon_edge = -P.ReduceMean()(P.ReduceSum()(
            P.Log()(P.ClipByValue()(beta_edge_s, 1e-10, np.inf) / P.ClipByValue()(z_edge_w, 1e-10, np.inf)), -1), 0)

        # B,ns -> B

        if not self.args.eval and self.args.prior_type in ['Dir2',
                                                           'Gaussian'] and self.args.iter_ < self.args.iter_threahold:
            loss = recon_word + KL1
        else:
            loss = (recon_edge + recon_word + KL1 + KL2 + recon_structure)  #

        if self.args.use_td:
            td_loss = self.tdregular()
            loss = loss + self.args.td_ratio * td_loss
        if self.args.use_recon:
            gcn_loss = self.loss_recon(gcn_recon, gcn_init)
            loss = loss + self.args.gcn_ratio * gcn_loss
        if self.args.use_mr:
            mr_loss = self.mr_regular(doc_g, theta, batch_doc_ids)
            loss = loss + self.args.mr_ratio * mr_loss


        if P.ReduceSum()(P.IsNan()(loss)) > 0 or P.ReduceMean()(loss) > 1e20 or P.ReduceSum()(
                P.IsNan()(recon_structure)) > 0:
            ipdb.set_trace()

        outputs = {
            "loss": loss.mean(),
            "recon_word": recon_word.mean(),
            "recon_edgew": recon_edge.mean(),
            "recon_structure": recon_structure.mean(),
            "p_edge": p_edge.mean(),
            "KL1": KL1.mean(),
            "KL2": KL2.mean()
        }
        if self.args.use_td:
            outputs.update({'td_reg': td_loss.mean()})
        if self.args.use_recon:
            outputs.update({'gcn_loss': gcn_loss.mean()})
        if self.args.use_mr:
            outputs.update({'mr_loss': mr_loss.mean()})

        return outputs

    def reset_parameters(self):
        self.topic_linear.weight = Parameter(
            Tensor(np.random.normal(0, 0.01, [1, 2 * self.args.ni]), mindspore.float32))
        # nn.init.constant_(self.topic_linear.weight, val=0)
        # nn.init.constant_(self.topic_linear.bias, val=0)
        pass

        # @snoop
    def loss(self, batch_data, gcn_inputs_batch, gcn_idxs_batch, idx2sent_batch, bow_batch, batch_doc_ids):
        return self.construct(batch_data, gcn_inputs_batch, gcn_idxs_batch, idx2sent_batch, bow_batch, batch_doc_ids)

    def get_beta(self):
        matmul = P.MatMul()  # 矩阵乘法操作
        softmax = P.Softmax(axis=-1)  # Softmax操作
        beta = matmul(self.topic_vec, self.word_vec_beta.T)
        beta = softmax(beta)

        return beta

    def get_W(self):
        concat = P.Concat(axis=-1)
        chunk = P.Split(axis=-1, output_num=2)
        matmul = P.MatMul()
        sigmoid = P.Sigmoid()
        eye = P.Eye()
        subtract = P.Sub()
        clamp = P.Clamp(min=1e-4, max=1 - 1e-4)
        tew_vector = concat((self.topic_vec, self.topic_edge_vec))
        topic_vec = self.topic_linear(tew_vector)
        head_vec, tail_vec = chunk(topic_vec)
        head_vec = L2_norm(head_vec)
        tail_vec = L2_norm(tail_vec)
        W = matmul(head_vec, tail_vec.T)
        W = sigmoid(4 * W)
        I = eye(self.args.num_topic)
        mask = subtract(1, I)

        return clamp(W)

    def get_beta_edge(self, weight=True):
        concat = P.Concat(axis=-1)
        prod = P.ReduceProd()
        matmul = P.MatMul()
        softmax = P.Softmax(axis=-1)

        beta = self.get_beta()
        beta_edge_w = beta[:, self.whole_edge]
        beta_edge_w = prod(beta_edge_w, -1)
        beta_nedge_w = 1 - beta_edge_w.sum(-1)
        beta_edge_w = concat((beta_nedge_w.unsqueeze(1), beta_edge_w))

        edge_vec = self.word_vec_beta[self.whole_edge, :]
        edge_vec = concat((edge_vec[0], edge_vec[1]))
        edge_vec = concat((self.noedge_vec, edge_vec))

        beta_edge = matmul(self.topic_edge_vec, edge_vec.T)
        beta_edge = weightedSoftmax(beta_edge, beta_edge_w)

        if weight:
            beta_edge = beta_edge * beta_edge_w

        return beta_edge

    def get_degree(self, weight=True):
        beta_edge = self.get_beta_edge(weight).permute(1, 0)  ##(1 + edge_size, K)
        beta_matrix = SparseTensor(row=self.whole_edge[0], col=self.whole_edge[1],
                                   value=beta_edge[1:, :],    
                                   sparse_sizes=(self.args.vocab_size + 1, self.args.vocab_size + 1))    # vocab_size, vocab_size, K
        out_degree = beta_matrix.sum(0)  # vocab_size *K
        in_degree = beta_matrix.sum(1)  # vocab_size *K
        degree = out_degree + in_degree  # vocab_size *K
        degree = degree.permute(1, 0)  # K * vocab_size

        return degree

    def get_doctopic(self, batch_data, batch_gcn_inputs, batch_gcn_idxs, batch_idx2sent):
        idx_x = batch_data.x
        x_batch = batch_data.x_batch
        idx_w = batch_data.x_w
        edge_w = batch_data.edge_w
        edge_index = batch_data.edge_index 
        param, phi, _, _ = self.encoder(idx_x, idx_w, x_batch, edge_index, edge_w, batch_gcn_inputs, batch_gcn_idxs, batch_idx2sent)  # (B, K ) (B*len, K) B*len = B*len(idx_x)
        Ns = myscattersum(torch.ones_like(idx_w, device=idx_w.device, dtype=torch.float), x_batch) #B
        phis = myscattersum(phi * torch.unsqueeze(idx_w, dim=-1), x_batch) #B*K
        theta = phis / torch.unsqueeze(Ns, dim=-1)
        # theta = self.encoder.reparameterize(param)
        return theta  # (B, K )

    def tdregular(self):
        sqrt = P.Sqrt()
        sum = P.ReduceSum()
        asin = P.Asin()
        det = P.Det()

        beta = self.get_beta()
        topic_vec = L2_norm(beta)
        cos = sum(topic_vec.unsqueeze(0) * topic_vec.unsqueeze(1), -1)
        mean = asin(sqrt(det(cos)))
        var = (math.pi / 2 - sqrt(det(cos))) ** 2

        return mean - var

    def mr_regular(self, bow_batch, theta, batch_doc_ids):
        pow = P.Pow()
        sum = P.ReduceSum()

        loss = Tensor(0.)
        for i in range(len(bow_batch)):
            for key in list(bow_batch[i].keys()):
                if key in batch_doc_ids and not batch_doc_ids.index(key) == i:
                    loss = loss + sum(pow(theta[i] - theta[batch_doc_ids.index(key)], 2))

        return loss / 2

    def loss_recon(self, recon_x, x):
        cosine_similarity = P.CosineSimilarity(axis=1)
        mean = P.ReduceMean()

        cosine_similarity_value = 1.0 - cosine_similarity(recon_x, x)
        cosine_loss = mean(cosine_similarity_value)

        return cosine_loss


def L2_norm(vec):
    sqrt = P.Sqrt()
    sum = P.ReduceSum()
    div = P.Div()

    nvec = sqrt(sum(vec**2, -1))
    return div(vec, nvec + 1e-10)

def weightedSoftmax(x, w):
    log = P.Log()
    exp = P.Exp()
    logsumexp = P.ReduceLogSumExp(axis=-1)
    sub = P.Sub()

    logwsum = logsumexp(x + log(w + 1e-20))
    return exp(sub(x, logwsum))

def to_BOW(idx, idx_w, idx_batch, V):
    eye = P.Eye()
    mul = P.Mul()
    scatter = P.ScatterAdd()

    embeddings = eye(V)
    batch_data = embeddings[idx]
    batch_data = mul(idx_w.unsqueeze(1), batch_data)
    size = int(idx_batch.max().item() + 1)
    batch_data = scatter(batch_data, idx_batch, size)

    return batch_data


def hard_z(y_soft, dim):
    argmax = P.ArgMaxWithValue(axis=dim)
    scatter = P.ScatterNd()
    sub = P.Sub()
    add = P.TensorAdd()

    index = argmax(y_soft)[0].unsqueeze(dim)
    y_hard = scatter(index, Tensor(1.0, y_soft.dtype), y_soft.shape)
    ret = add(sub(y_hard, y_soft), y_soft)

    return ret


def myscattersum(src, index, size=None):
    scatter_add = P.ScatterAdd()

    if src is None:
        src = Tensor(1.0, index.dtype)

    flag = (index[1:] - index[:-1]).astype('int32')
    flag = P.Concat()([flag, Tensor([1])])
    RANGE = P.OnesLike()(index).cumsum(0) - 1

    ids = P.Where()(flag > 0)[0]
    nids = ids[1:] - ids[:-1]
    nids = P.Concat()([ids[[0]] + 1, nids])
    ids = ids[:-1] + 1

    yindex = P.ZerosLike()(index)
    yindex[ids] = nids[:-1]
    yindex = RANGE - yindex.cumsum(0)

    indexs = P.Stack()([index, yindex])
    ST = P.SparseTensor(indexs, src)
    ret = ST.to_dense().sum(1)

    if size is not None and size > ret.shape[0]:
        N = size - ret.shape[0]
        zerovec = Tensor(N, *ret.shape[1:])
        ret = P.Concat(0)([ret, zerovec])

    return ret


def adj_mask(x_batch):
    max_op = P.ReduceMax()
    zeros_op = P.ZerosLike()
    ones_op = P.OnesLike()
    where_op = P.Where()

    size = max_op(x_batch)
    N = x_batch.shape[0]

    mask2 = zeros_op(x_batch)

    for i in range(size + 1):
        idxs = where_op(x_batch == i)[0]
        mask2[idxs[0]:idxs[-1] + 1, idxs] = ones_op(idxs)

    mask2_diag_embedded = Tensor(np.eye(N))

    mask2 = mask2 - mask2_diag_embedded

    mask = ones_op(mask2) - mask2

    return mask, mask2
class RelaxedOneHotCategorical(nn.Cell):
    def __init__(self, temperature, probs):
        super(RelaxedOneHotCategorical, self).__init__()
        self.temperature = temperature
        self.probs = probs
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.gumbel = ops.GumbelSample()

    def construct(self):
        gumbel_noise = self.gumbel(Tensor(np.zeros_like(self.probs.asnumpy())))
        sample = (self.log_softmax(self.probs) + gumbel_noise) / self.temperature
        return nn.Softmax(axis=-1)(sample)

if __name__ == '__main__':
    # import torch
    #
    # x = torch.rand(14, 3)
    # # x_batch = torch.zeros(10)
    # # ids = torch.randint(0, 10, [100])
    # # x_batch[ids] = 1
    # # x_batch = x_batch.cumsum(dim=-1).long()
    # x_batch = torch.tensor([0,0,0,1,1,1,2,2,4,4,4,5,5,5])
    # ret = myscattersum(x, x_batch)
    # x = Tensor(mindspore.common.initializer('normal', shape=[14, 3]))
    x = ops.normal((14, 3))
    x_batch = Tensor([0, 0, 0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5])

    # 这里假设你已经定义了一个名为myscattersum的函数
    ret = myscattersum(x, x_batch)
    import ipdb
    ipdb.set_trace()
