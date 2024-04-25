import mindspore as ms
import mindcv
from mindspore import nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, TruncatedNormal, Constant
from .layers import DropPath, Identity, Mlp
from .resnet import *

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj.weight.set_data(initializer(0.0, self.proj.weight.shape, self.proj.weight.dtype))
        self.proj.bias.set_data(initializer(0.0, self.proj.bias.shape, self.proj.bias.dtype))

    def construct(self, q, k, v):
        B_q, N_q, C_q = q.shape
        B_k, N_k, _ = k.shape
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1)
        q = ops.Transpose()(q, (0, 2, 1, 3))
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1)
        k = ops.Transpose()(k, (0, 2, 1, 3))
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1)
        v = ops.Transpose()(v, (0, 2, 1, 3))
        attn = ops.matmul(q, ops.Transpose()(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        q = ops.matmul(attn, v)
        q = ops.Transpose()(q, (0, 2, 1, 3)).reshape(B_q, N_q, C_q)
        # q = (attn @ v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        q = self.proj_drop(self.proj(q))
        return q


class Decoder(nn.Cell):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm((dim,))
        self.bn2 = nn.LayerNorm((dim,))
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def construct(self, q, x):
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
        return q

class VSA_Module(nn.Cell):
    def __init__(self, hash_bit):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = 3
        out_channels = 4
        embed_dim = hash_bit  # opt['embed']['embed_dim']

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=3, stride=4)  # 192 768
        self.HF_conv = nn.Conv2d(in_channels=3072, out_channels=channel_size, kernel_size=1, stride=1)  # 768 3072
        # visual attention
        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size * 2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size * 2, out_channels=out_channels, kernel_size=1)

        # solo attention
        # self.decoder = nn.ModuleList([Decoder(32, 8, True, 0.1, 0.1, 0.1) for _ in range(2)])
        self.decoder = Decoder(32, 4, True, 0.1, 0.1, 0.1)
        self.proj = nn.SequentialCell(nn.Dense(in_channels=32 * 4, out_channels=hash_bit))

    def construct(self, lower_feature, higher_feature, solo_feature):
        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)
        #print(lower_feature.shape,higher_feature.shape)
        # concat
        concat_feature = ops.cat((lower_feature, higher_feature), axis=1)

        # residual
        concat_feature = ops.mean(higher_feature, axis=1, keep_dims=True).expand_as(concat_feature) + concat_feature

        # attention
        main_feature = self.conv1x1_1(concat_feature)

        atten2 = self.conv1x1_2(concat_feature)  # .view(concat_feature.shape[0],1,-1)
        x = (main_feature * ops.sigmoid(atten2)).reshape(atten2.shape[0], 32, 32)
        x = self.decoder(solo_feature.reshape(x.shape[0], 4, 32), x)
        solo_feature = self.proj(x.reshape(x.shape[0], -1))
        return solo_feature

class ExtractFeature(nn.Cell):
    def __init__(self, pretrained=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = 64
        self.pretrained = pretrained
        # self.resnet = mindcv.create_model('resnet50', pretrained=self.pretrained)

        self.resnet = resnet50(pretrained=self.pretrained, model_file="./resnet50.ckpt")
        self.pool_2x2 = nn.MaxPool2d(4)
        self.linear = nn.Dense(in_channels=2048 * self.embed_dim, out_channels=4 * 32)

    def construct(self, img):
        feature_list = self.resnet(img)
        f1, f2, f3, f4 = feature_list
        # print("net: ", f1.shape, f2.shape, f3.shape, f4.shape)
        # Lower Feature
        f2_up = ops.interpolate(f2, size=(f1.shape[2], f1.shape[3]), mode='nearest')
        lower_feature = ops.cat((f1, f2_up), axis=1)

        # Higher Feature
        f4_up = ops.interpolate(f4, size=(f3.shape[2], f3.shape[3]), mode='nearest')
        higher_feature = ops.cat((f3, f4_up), axis=1)
        # higher_feature = self.up_sample_4(higher_feature)

        # batch * 512
        feature = f4.view(f4.shape[0], -1)
        solo_feature = self.linear(feature)

        return lower_feature, higher_feature, solo_feature


class Model(nn.Cell):
    def __init__(self, hash_bit, pretrained=True):
        super(Model, self).__init__()
        self.bit = hash_bit
        self.pretrained = pretrained
        self.model1 = ExtractFeature(pretrained=self.pretrained)
        self.model2 = VSA_Module(hash_bit)
        # self.cls = nn.Dense(hash_bit, 25)

    def construct(self, img, label=None):
        lower_feature, higher_feature, solo_feature = self.model1(img)
        solo_feature = self.model2(lower_feature, higher_feature, solo_feature)
        if label == None:  # Test Only
            solo_feature = ops.L2Normalize(axis=-1)(solo_feature)
            return solo_feature

        return solo_feature

    # def construct(self, img, label=None):
    #     lower_feature, higher_feature, solo_feature = self.model1(img)
    #     solo_feature = self.model2(lower_feature, higher_feature, solo_feature)
    #     if label == None:  # Test Only
    #         solo_feature = ops.L2Normalize(axis=-1)(solo_feature)
    #         return solo_feature
    #
    #     # Loss
    #     loss1 = (solo_feature.abs() - 1).pow(2).abs()
    #     loss1 = loss1.mean()
    #
    #     u = ops.tanh(solo_feature)
    #     dist = (u.unsqueeze(1) - u.unsqueeze(0)).pow(2).sum(dim=2)
    #     s = (label @ label.t() == 0).float()
    #
    #     ld = (1 - s) / 2 * dist + s / 2 * (2 * self.bit - dist).clamp(min=0)
    #     ld = ld.mean()
    #     loss = self.cls(solo_feature)
    #     loss = ops.cross_entropy(loss, label.argmax(1))
    #     return solo_feature, loss + 0.1 * loss1 + ld

class NetWithLossCell(nn.Cell):
    '''NetWithLossCell'''
    def __init__(self, network, hash_bit=64):
        super(NetWithLossCell, self).__init__()
        self.network = network
        self.cls = nn.Dense(hash_bit, 25)
        self.bit = hash_bit


    def construct(self, img, label):
        # Loss
        solo_feature = self.network(img, label)
        loss1 = (solo_feature.abs() - 1).pow(2).abs()
        loss1 = loss1.mean()

        u = ops.tanh(solo_feature)
        dist = u.unsqueeze(1) - u.unsqueeze(0)
        dist = ops.pow(dist, 2)
        dist = ops.sum(dist, dim=2)
        label = ops.Cast()(label, ms.float32)  # lb (128, 25)
        label_t = ops.Transpose()(label, (1, 0))  # lb (25, 128)
        s = ops.Cast()(ops.matmul(label, label_t), ms.bool_)
        s = ops.logical_not(s)
        s = ops.Cast()(s, ms.float32)
        ld = (1 - s) / 2 * dist + s / 2 * (2 * self.bit - dist).clamp(min=0)
        ld = ld.mean()
        loss = self.cls(solo_feature)
        loss = ops.cross_entropy(loss, label.argmax(1))
        loss = loss + 0.1 * loss1 + ld
        return loss


if __name__ == "__main__":
    image = ms.ops.randn(50, 3, 224, 224)
    network = Model(hash_bit=64, pretrained=True)
    # output, loss = network(image, label)
    print(network)
    # print(f"network output shape: {network(dummy_input).shape}")