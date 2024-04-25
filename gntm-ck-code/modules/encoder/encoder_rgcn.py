# import torch
import mindspore
import torch
from mindspore import load_checkpoint, load_param_into_net,Tensor,Parameter
import numpy as np
# import torch.nn as nn
# from torch_scatter import scatter
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import scatter
# from torch.autograd import Variable
import pickle
from mindspore import  Tensor
from mindspore.nn import Dense,Dropout,BatchNorm1d
from rgcn import RGCN
from mindspore.ops import operations as P
from mindspore.train.serialization import save_checkpoint, load_checkpoint
import pdb
from mindspore import ops
from mindspore import Tensor, dtype as mstype
import sys
from mindspore.common.initializer import initializer
# import torchsnooper
# torchsnooper.register_snoop()
from mindspore import  context
context.set_context(max_call_depth=2000)
context.set_context(mode=context.PYNATIVE_MODE)
def scatter_sum(src, index, dim_size):
    scatter = P.ScatterAdd()
    output = mindspore.ops.zeros(dim_size, mindspore.float32)
    return scatter(output, index, src)
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
class RGCNDir2encoder(nn.Cell):

    def __init__(self, args, word_vec):
        super(RGCNDir2encoder, self).__init__()
        self.tanh = ops.Tanh()
        self.isnan = ops.IsNan()
        self.concat = ops.Concat(axis=-1)
        self.sigmoid = ops.Sigmoid()
        self.scatter = P.ScatterNdUpdate()
        self.softmax = ops.Softmax(axis=-1)
        self.args= args
        if word_vec is not None and args.word:
            self.word_vec = word_vec
            if args.fixing:
                self.word_vec.requires_grad = False
        else:
            # self.word_vec = Tensor(np.eye(args.vocab_size, dtype=np.float32), mindspore.float64)
            self.word_vec = ops.eye(args.vocab_size, dtype=mindspore.float32)
        input_size = args.nw

        relation_map = pickle.load(open(args.path + "relation_map_path%d.pkl" % args.num_path, 'rb'))
        unique_nodes_mapping = pickle.load(open(args.path + "unique_nodes_mapping_path%d.pkl" % args.num_path, 'rb'))
        print("before RGCN")
        gcn_model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=4, dropout=0.0)
        print("after RGCN")
        gcn_save_path =  args.path + "weights_path%d/model_epoch%d" % (args.num_path, args.gcn_epoch) + ".pt"

        torch_params=torch.load(gcn_save_path)
        torch_params = {k: v.cpu().numpy() for k, v in torch_params.items()}

        # 加载参数到MindSpore模型
        for name, param in gcn_model.parameters_dict().items():
            if name in torch_params:
                param.set_data(Tensor(torch_params[name]))

        self.enc1_gnn1 = gcn_model

        self.gcn_proj = nn.SequentialCell(
            nn.Dense(args.g_dim, args.nw),
            nn.Dense(args.nw, args.nw * 2),
            nn.LeakyReLU(),
            nn.Dense(args.nw * 2, args.nw)
            )
        if args.use_recon:
            print('reconlayer   runhere')
            self.recon_layer = nn.SequentialCell([nn.Dense(args.nw, 100), nn.Dense(100, args.g_dim)])

        self.bn_gnn1 = nn.BatchNorm1d(args.nw)

        self.enc2_fc1 = nn.Dense(input_size + args.nw, args.enc_nh)   # f1
        self.enc2_fc2 = nn.Dense(input_size + args.nw, args.enc_nh)   # f2
        self.enc2_drop = nn.Dropout(0.2)

        self.mean_fc = nn.Dense(args.enc_nh, args.num_topic)  # 100  -> 50  # f_miu
        self.mean_bn = nn.BatchNorm1d(args.num_topic)  # bn for mean
        self.logvar_fc = nn.Dense(args.enc_nh, args.num_topic)  # 100  -> 50   # f_sigma
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)  # bn for logvar

        self.phi_fc = nn.Dense(args.nw + input_size + args.enc_nh, args.num_topic)  # f_phi
        self.phi_bn = nn.BatchNorm1d(args.num_topic)

        self.logvar_bn.weight = Parameter(Tensor(np.ones([args.num_topic]), mindspore.float32), requires_grad=False)
        self.mean_bn.weight = Parameter(Tensor(np.ones([args.num_topic]), mindspore.float32), requires_grad=False)
        prior_mean = Tensor(np.zeros([1, args.num_topic]), mindspore.float32)
        prior_var = Tensor(np.ones([1, args.num_topic]) * args.variance, mindspore.float32)
        prior_logvar = P.Log()(prior_var)

        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_logvar = prior_logvar

        self.reset_parameters()

    def reset_parameters(self):
        self.logvar_fc.weight = Parameter(Tensor(np.zeros([self.args.num_topic, self.args.enc_nh]), mindspore.float32))
        self.logvar_fc.bias = Parameter(Tensor(np.zeros([self.args.num_topic]), mindspore.float32))
        pass

    # @snoop
    # def construct(self, idx_x, idx_w, x_batch, edge_index, edge_weight, batch_gcn_inputs, batch_gcn_idxs,
    #                 batch_idx2sent):
    def construct(self, idx_x, idx_w, x_batch, edge_index, edge_weight, batch_gcn_inputs, batch_gcn_idxs, batch_idx2sent):
        x = self.word_vec[idx_x]
        gcn_embeds = []
        for i, g in enumerate(batch_gcn_inputs):
            print("before rcgn")
            # 得到每个节点的embedding
            feature_emb = self.enc1_gnn1(g.entity, g.edge_index, g.edge_type, g.edge_norm)
            print("after rcgn")
            # 因为g中对文档中的单词顺序做出了改变，所以
            # 要得到原始文档中每个单词对应的embedding需要
            # 对应到文档中每个单词对应的次序(存放在batch_gcn_idxs中)
            lst = [i for i in batch_gcn_idxs[i] if 0 <= i < len(feature_emb)]
            feature_emb = feature_emb[lst]
            gcn_embeds.append(feature_emb)

        # for i in range(len(entity_list)):
        #     print(" RGCNDir2 construct",entity_list[i])
        #     feature_emb = self.enc1_gnn1(entity_list[i], edge_d_index[i], edge_type[i], edge_norm[i])
        #     feature_emb = feature_emb[batch_gcn_idxs[i]]
        #     gcn_embeds.append(feature_emb)
        # 这一步是为了对齐x中每个单词的顺序
        # 防止第138行中的代码拼接不同单词的表示 - enc1 = torch.cat([enc1, x], dim=-1)
        # 也就是说enc1和x中的单词顺序是不一样的，需要根据flat_idx2sent对应起来
        flat_idx2sent = []
        accum_sum = 0
        for i in range(len(batch_idx2sent)):
            if i == 0:
                flat_idx2sent.extend(batch_idx2sent[i])
            else:
                accum_sum += len(batch_idx2sent[i - 1])

        gcn_init_output = P.Concat(axis=0)(gcn_embeds)  # B*len, embed
        #
        # 应用gcn_proj
        cast = ops.Cast()
        gcn_init_output = cast(gcn_init_output, mindspore.float32)

        gcn_output = self.gcn_proj(gcn_init_output)  # B*len, embed

        # 在训练阶段使用recon_layer
        if self.args.use_recon:
            gcn_recon_output = self.recon_layer(gcn_output)  # shape=[B, max_len, 100]
        else:
            gcn_recon_output = None

        # 将Tensor转换为NumPy数组
        gcn_output_np = gcn_output.asnumpy()

        # 使用整数列表来索引NumPy数组
        gcn_output_np = gcn_output_np[flat_idx2sent]

        # 将NumPy数组转换回Tensor
        gcn_output = Tensor(gcn_output_np)
        cast2 = P.Cast()
        gcn_output = cast2(gcn_output, mindspore.float32)

        #=======================================================================================================

        enc1 = self.tanh(gcn_output)
        if self.isnan(enc1).sum() > 0:
            print("NaN values detected in enc1")

        enc1 = self.concat((enc1, x))
        enc2 = self.sigmoid(self.enc2_fc1(enc1)) * self.tanh(self.enc2_fc2(enc1))
        size = int(x_batch.max().item() + 1)
        enc2 = scatter(enc2, 0, x_batch, enc2)  # (B, ns,K)
        enc2d = self.enc2_drop(enc2)
        mean = self.mean_bn(self.mean_fc(enc2d))  # posterior mean
        enc2d = enc2d.astype(mindspore.float64)
        logvar = self.logvar_fc(enc2d)  # posterior log variance
        word_embed = self.concat((enc1, enc2[x_batch]))
        phi = self.softmax(self.phi_fc(word_embed))  # (B*max_len)*num_topic



        param = (mean, logvar)
        # 检查mean中是否有NaN值
        if P.ReduceSum()(P.IsNan()(mean)).asnumpy().item() > 0:
            print("NaN values detected in mean")

        return param, phi, gcn_recon_output, gcn_init_output

    def reparameterize(self, param):
        posterior_mean = param[0]
        posterior_var = param[1].exp()
        # take sample
        if self.training:
            eps = mindspore.ops.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        else:
            z = posterior_mean
        theta = mindspore.ops.softmax(z, axis=-1)
        return theta

    def KL_loss(self, param):    
        posterior_mean = param[0]
        posterior_logvar = param[1]

        # prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        # prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        # prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        prior_mean = Tensor(self.prior_mean).expand_as(posterior_mean)
        prior_var = Tensor(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Tensor(self.prior_logvar).expand_as(posterior_mean)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        KL = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.args.num_topic)

        if mindspore.ops.isinf(KL).sum() > 0 or mindspore.ops.isnan(KL).sum() > 0:
            pdb.set_trace()

        return KL
