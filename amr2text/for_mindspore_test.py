from mindspore.train.serialization import save_checkpoint

from mindspore import Tensor
import mindspore
from itertools import zip_longest
import torch

def pytorch2mindspore(ckpt_name='to_convert_model.pth'):

    par_dict = torch.load(ckpt_name, map_location=torch.device('cpu'))

    new_params_list = []

    for name in par_dict["model"]:
        param_dict = {}
        parameter = par_dict["model"][name]
        print(name, parameter)
        print('========================py_name',name)
        if name.endswith('emb_luts.0.weight'):

            name = name[:name.rfind('emb_luts.0.weight')]

            name = name + '0.0.embedding_table'

        print('========================ms_name',name)
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())

        new_params_list.append(param_dict)

    for name in par_dict["generator"]:
        param_dict = {}
        parameter = par_dict["generator"][name]
        print(name, parameter)
        print('========================py_name',name)
        name = 'generator.' + name
        print('========================ms_name',name)
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())

        new_params_list.append(param_dict)
    save_checkpoint(new_params_list,  'ms_amr2text.ckpt')

pytorch2mindspore()

# param_dict = mindspore.load_checkpoint("./model.ckpt")
# print(param_dict)
# par_dict = torch.load("to_convert_model.pth", map_location=torch.device('cpu'))
# print(par_dict.keys())
# torch_dict = par_dict['model']
# print(par_dict['generator']["0.weight"])
# print(len(par_dict["model"]))
# exit(0)
# # print(par_dict.keys())
# for ms_p, torch_p in zip_longest(param_dict, torch_dict):
#     if(ms_p != torch_p):
#         print(str(ms_p) + " " + str(torch_p))
    # print("ms: ", ms_p)
    # print("torch_p", torch_p)

# param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
# print(param_not_load)

# encoder.embeddings.make_embedding.0.0.embedding_table encoder.embeddings.make_embedding.emb_luts.0.weight
# decoder.embeddings.make_embedding.0.0.embedding_table decoder.embeddings.make_embedding.emb_luts.0.weight


# def pytorch2mindspore(ckpt_name='to_convert_model.pth'):
#
#     par_dict = torch.load(ckpt_name, map_location=torch.device('cpu'))
#
#     new_params_list = []
#
#     for name in par_dict["model"]:
#
#         param_dict = {}
#
#         parameter = par_dict[name]
#         print(name, parameter)
#
#         print('========================py_name',name)
#
#         if name.endswith('emb_luts.0.weight'):
#
#             name = name[:name.rfind('emb_luts.0.weight')]
#
#             name = name + '0.0.embedding_table'
#
#         # elif name.endswith('normalize.weight'):
#         #
#         #     name = name[:name.rfind('normalize.weight')]
#         #
#         #     name = name + 'normalize.gamma'
#         #
#         # elif name.endswith('.running_mean'):
#         #
#         #     name = name[:name.rfind('.running_mean')]
#         #
#         #     name = name + '.moving_mean'
#         #
#         # elif name.endswith('.running_var'):
#         #
#         #     name = name[:name.rfind('.running_var')]
#         #
#         #     name = name + '.moving_variance'
#
#         print('========================ms_name',name)
#
#         param_dict['name'] = name
#
#         param_dict['data'] = Tensor(parameter.numpy())
#
#         new_params_list.append(param_dict)

    # save_checkpoint(new_params_list,  'ms_amr2text.ckpt')