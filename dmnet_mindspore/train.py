import os
import time
from DMNet import DMNet
import get_data
from mindspore import nn, ops
import mindspore as ms
from mindspore import value_and_grad, jit, save_checkpoint, load_checkpoint, load_param_into_net, Tensor, Parameter

import utils

def train():
    opt = utils.get_global_config()

    # print('-------------------------------------Initial Data Object-------------------------------------')

    ## get data generator
    dp = get_data.Data_Prepare('train')
    val_dp = get_data.Data_Prepare('val')
    dataset = dp.get_loader()
    val_dataset = val_dp.get_loader()

    ## initial model
    assert opt['model_name'] == 'diver'
    model = DMNet()
    if(os.path.exists(opt['model_dir'])):
        param_dict = load_checkpoint(opt['model_dir'])
        load_param_into_net(model, param_dict)

    print('---------------------Model Parameter Configuration--------------------------')
    for name, param in model.parameters_and_names():
        print(f"Layer: {name:{30}}  Trainable: {param.requires_grad:{1}}  Size: {param.shape}")

    ## initial loss_function
    assert opt['loss'] == 'cce'
    class CCEloss(nn.Cell):
        def construct(self, out, labels):
            out = out.view(opt['way']*opt['query'], opt['way'])
            print(1-out)
            loss_function = nn.CrossEntropyLoss()
            return loss_function(1-out, labels)

    loss_fn = CCEloss()

    ## initial optimizer
    assert opt['optim_method'] == 'Adam'

    class Momentum(nn.Optimizer):
        def __init__(self, params, learning_rate=0.001, momentum=0.9):
            super(Momentum, self).__init__(learning_rate, params)
            self.momentum = Parameter(Tensor(momentum, ms.float32), name="momentum")
            self.moments = self.parameters.clone(prefix="Momentum", init="zeros")

        def construct(self, gradients):
            lr = self.get_lr()
            params = self.parameters
            for i in range(len(params)):
                ops.assign(self.moments[i], self.moments[i] * self.momentum + gradients[i])
                update = params[i] - self.moments[i] * lr
                ops.assign(params[i], update)
            return params
        

    # optim = nn.Adam(model.trainable_params(), learning_rate=opt['learning_rate'], weight_decay=opt['weight_decay'], eps=1e-4)
    optim = Momentum(model.trainable_params(), 0.001)
    # optim = nn.Momentum()
    # optim = nn.SGD()


    ## define network
    def net_forward(input, target):
        out = model(input)
        loss = loss_fn(out, target)
        return loss, out

    net_backword = value_and_grad(net_forward, None, weights=optim.parameters)

    @jit
    def train_step(input, target):
        (loss, out), gard = net_backword(input, target)
        optim(gard)
        acc = utils.cal_acc(out, target[opt['way']*opt['shot']:])
        return loss, acc
    
    @jit
    def vaild_step(input, target):
        (loss, out), _ = net_backword(input, target)
        acc = utils.cal_acc(out, target[opt['way']*opt['shot']:])
        return loss, acc

    print('-------------------------------------Start training-------------------------------------')
    best_acc = 0.0
    for i in range(1, opt['train_episodes']+1):

        t = time.time()
        print(f'Training... current episode: {i}', end=' ')

        for (data, label) in dataset.create_tuple_iterator():
            #adjust label
            label = label.asnumpy()
            label_dict = {}
            for i in range(opt['way']):
                label_dict[label[i*opt['shot']]] = i
            label = label[opt['way']*opt['shot']:]
            real_label = [label_dict[i] for i in label]
            real_label = Tensor(real_label).to(ms.int32)
            loss, acc = train_step(data, real_label)
            print(f'train_loss is {loss:}   train_acc is {acc:}   time cost is {time.time()-t:0.6f} s')

        if(i%opt['decay_every'] == 0): # decay learning rate
            ops.assign(optim.learning_rate, opt['learning_rate']*0.5)
            opt['learning_rate']*=0.5

        if(i%1000 == 0): # valid once per 1000 episode

            print(f'Validing... current episode: {i}')
            test_acc = 0.0
            test_loss = 0.0

            for j in range(1, opt['test_episodes']+1):
                print(f'In validing {i//1000} ,episode: {j}', end=' ')

                for (data, label) in val_dataset.create_tuple_iterator():
                    label = label.asnumpy()
                    label_dict = {}
                    for i in range(opt['way']):
                        label_dict[label[i*opt['shot']]] = i
                    label = label[opt['way']*opt['shot']:]
                    real_label = [label_dict[i] for i in label]
                    real_label = Tensor(real_label).to(ms.int32)
                    loss, acc = vaild_step(data, real_label)
                    print(f'train_loss is {loss}   train_acc is {acc}   time cost is {time.time()-t:0.6f} s')
                    test_acc += acc
                    test_loss += loss

            test_acc /= opt['test_episodes']
            test_loss /= opt['test_episodes']
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'==> best model (acc = {test_acc}, loss = {test_loss}), saving model...')
                save_checkpoint(model, opt['model_dir'])