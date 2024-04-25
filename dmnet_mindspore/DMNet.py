import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from model_helper import *
import utils
# from Model.utils import feature_sum

class DMNet(nn.Cell):

    def __init__(self):
        super().__init__()
        self.opt = utils.get_global_config()
        self.ops = self.opt['diver_ops']
        self.way = self.opt['way']
        self.shot = self.opt['shot']
        self.query = self.opt['query']
        self.layer_num = self.opt['layer_num']
        self.origin_channel = self.opt['orain_dim']
        self.out_channel = self.opt['enc_channel']
        print('self.layer_num', self.layer_num)

        if self.layer_num == 10:
            #part1
            self.conv_1 = myconvpad(self.origin_channel, self.out_channel)            # 3*84->64*84
            self.conv_2 = myconvpad_pooling(self.out_channel, self.out_channel)      # 42
            self.conv_3 = myconvpad(self.out_channel, self.out_channel)              # 42
            self.conv_4 = myconvpad_pooling(self.out_channel, self.out_channel)      # 21
            self.conv_5 = myconvpad(self.out_channel, self.out_channel)              # 21
            self.conv_6 = myconvpad(self.out_channel, self.out_channel)              # 21
            self.conv_7 = myconvpad_pooling(self.out_channel, self.out_channel)      # 10
            self.conv_8 = myconvpad(self.out_channel, self.out_channel)              # 10
            self.conv_9 = myconvpad(self.out_channel, self.out_channel)              # 10
            self.conv_10 = myconvpad_pooling(self.out_channel, self.out_channel)     # 5
            #part3
            self.basic_1 = myconvpad_pooling(self.out_channel, self.out_channel)
            self.basic_2 = myconvpad(2*self.out_channel, self.out_channel)
            self.basic_3 = myconvpad_pooling(2*self.out_channel, self.out_channel)
            self.basic_4 = myconvpad(2*self.out_channel, self.out_channel)
            self.basic_5 = myconvpad(2*self.out_channel, self.out_channel)
            self.basic_6 = myconvpad_pooling(2*self.out_channel, self.out_channel)
            self.basic_7 = myconvpad(2*self.out_channel, self.out_channel)
            self.basic_8 = myconvpad(2*self.out_channel, self.out_channel)
            self.basic_9 = myconvpad_pooling(2*self.out_channel, self.out_channel)
        else:
            self.conv_1 = myconvpad(self.origin_channel, self.out_channel)            # 3*84->64*84
            self.conv_2 = myconvpad_pooling(self.out_channel, self.out_channel)      # 42
            self.conv_4 = myconvpad_pooling(self.out_channel, self.out_channel)      # 21
            self.conv_7 = myconvpad_pooling(self.out_channel, self.out_channel)      # 10
            self.conv_10 = myconvpad_pooling(self.out_channel, self.out_channel)     # 5

            self.basic_1 = myconvpad_pooling(self.out_channel, self.out_channel)
            self.basic_3 = myconvpad_pooling(2*self.out_channel, self.out_channel)
            self.basic_6 = myconvpad_pooling(2*self.out_channel, self.out_channel)
            self.basic_9 = myconvpad_pooling(2*self.out_channel, self.out_channel)

        self.conv_block = myconvpad(2*self.out_channel, self.out_channel)
        self.flatten = Flatten
        self.fc_1 = nn.Dense(64*5*5, 32)
        self.fc_2 = nn.Dense(32, 1)

    # @ snoop
    def construct(self, x):

        #x: (n_s+n_q, C, H, W)
        if self.layer_num == 10:
            conv_1 = self.conv_1(x)
            conv_2 = self.conv_2(conv_1)
            conv_3 = self.conv_3(conv_2)
            conv_4 = self.conv_4(conv_3)
            conv_5 = self.conv_5(conv_4)
            conv_6 = self.conv_6(conv_5)
            conv_7 = self.conv_7(conv_6)
            conv_8 = self.conv_8(conv_7)
            conv_9 = self.conv_9(conv_8)
            conv_10 = self.conv_10(conv_9)

            encoder_features = [conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9, conv_10]
        else:
            conv_1 = self.conv_1(x)
            conv_2 = self.conv_2(conv_1)
            conv_3 = self.conv_4(conv_2)
            conv_4 = self.conv_7(conv_3)
            conv_5 = self.conv_10(conv_4)

            encoder_features = [conv_1, conv_2, conv_3, conv_4, conv_5]
        
        '''-------------------------------------------------------------------------------------------------------------------------------------------'''
        '''-------------------------------------------------------------------------------------------------------------------------------------------'''
        '''-------------------------------------------------------------------------------------------------------------------------------------------'''
        
        divers = []
        
        for k in range(len(encoder_features)):
            support_features = encoder_features[k][:self.way*self.shot,:,:,:]
            query_features = encoder_features[k][self.way*self.shot:,:,:,:]

            support_features_ext = support_features.view(self.way, self.shot, ops.shape(support_features)[-3], ops.shape(support_features)[-2], ops.shape(support_features)[-1])# (5, 5, C)

            class_ref = ops.mean(support_features_ext, 1)# (, 5, C)
            class_ref = ops.tile(ops.expand_dims(class_ref, 0), (int(self.way*self.query), 1, 1, 1, 1))# (50, 5, C) 

            query_features_ext = ops.tile(ops.expand_dims(query_features, 0), (self.way, 1, 1, 1, 1))# (5, 50, C)
            query_features_ext = query_features_ext.transpose((1,0,2,3,4))

            class_ref = ops.reshape(class_ref, (-1, ops.shape(class_ref)[-3], ops.shape(class_ref)[-2], ops.shape(class_ref)[-1]))
            query_features_ext = ops.reshape(query_features_ext, (-1, ops.shape(query_features_ext)[-3], ops.shape(query_features_ext)[-2], ops.shape(query_features_ext)[-1]))

            if self.ops=='add':
                cur_diver = query_features_ext + class_ref
            elif self.ops=='mul':
                cur_diver = query_features_ext * class_ref
            else:
                cur_diver = ops.abs(query_features_ext-class_ref)# 250, C, H, w

            divers.append(cur_diver)# layer, class*totalquery, C, H, W

        '''-------------------------------------------------------------------------------------------------------------------------------------------'''
        '''-------------------------------------------------------------------------------------------------------------------------------------------'''
        '''-------------------------------------------------------------------------------------------------------------------------------------------'''

        if self.layer_num == 10:
            conv_diver = self.basic_1(divers[0])
            concate_diver = ops.Concat(1)((divers[1], conv_diver))
            conv_diver = self.basic_2(concate_diver)
            concate_diver = ops.Concat(1)((divers[2], conv_diver))
            conv_diver = self.basic_3(concate_diver)
            concate_diver = ops.Concat(1)((divers[3], conv_diver))
            conv_diver = self.basic_4(concate_diver)
            concate_diver = ops.Concat(1)((divers[4], conv_diver))
            conv_diver = self.basic_5(concate_diver)
            concate_diver = ops.Concat(1)((divers[5], conv_diver))
            conv_diver = self.basic_6(concate_diver)
            concate_diver = ops.Concat(1)((divers[6], conv_diver))
            conv_diver = self.basic_7(concate_diver)
            concate_diver = ops.Concat(1)((divers[7], conv_diver))
            conv_diver = self.basic_8(concate_diver)
            concate_diver = ops.Concat(1)((divers[8], conv_diver))
            conv_diver = self.basic_9(concate_diver)
            concate_diver = ops.Concat(1)((divers[9], conv_diver))
        else:
            conv_diver = self.basic_1(divers[0])
            concate_diver = ops.Concat(1)((divers[1], conv_diver))
            conv_diver = self.basic_3(concate_diver)
            concate_diver = ops.Concat(1)((divers[2], conv_diver))
            conv_diver = self.basic_6(concate_diver)
            concate_diver = ops.Concat(1)((divers[3], conv_diver))
            conv_diver = self.basic_9(concate_diver)
            concate_diver = ops.Concat(1)((divers[4], conv_diver))

        final_conv_diver = self.conv_block(concate_diver)  #250,64,5,5

        flat = self.flatten(final_conv_diver) 
        out = self.fc_1(flat) #250,32
        out = self.fc_2(out) #250,1

        return out