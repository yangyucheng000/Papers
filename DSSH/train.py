import os
import argparse
import mindspore
from src.network import *
from src.data import *
from src.utils import *
import mindspore.nn as nn
from mindspore import context, set_seed
import time


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='DSSH')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='Device where the code will be implemented. (Default: GPU)')
    parser.add_argument('--epoch', type=int, default=500, help='Epoch size for train phase. (Default: 500)')
    parser.add_argument('--device_id', type=int, default=0, help='Device id. (Default: 0)')
    parser.add_argument('--do_shuffle', type=bool, default=True, choices=[True, False],
                        help='Enable shuffle for train dataset. (Default: true)')
    parser.add_argument('--output_dir', type=str, default='./output', help='The output checkpoint directory.')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Eval Batch size in callback')
    parser.add_argument('--bit_list', type=int, default=64, help='bit_list')
    parser.add_argument('--resume', type=bool, default=False, choices=[True, False], help='Resume')
    parser.add_argument('--eval_step', type=int, default=5, help='Eval Step')
    return parser.parse_args()


def train(args_opt):
    rank = 0
    device_num = 1
    bit_list = args_opt.bit_list
    ckpt_save_path = "./output/"

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    # Dataset
    train_dataset = create_dataset(phase="train", batch_size=args_opt.train_batch_size,
                                  device_num=device_num,
                                  rank=rank)
    dataset_size = train_dataset.get_dataset_size()
    print("train dataset size: ", dataset_size)
    eval_dataset = create_dataset(phase="eval", batch_size=args_opt.eval_batch_size,
                                  device_num=device_num,
                                  rank=rank)
    print("eval dataset size: ", eval_dataset.get_dataset_size())
    
    # Model
    f_model = Model(bit_list)
    model = NetWithLossCell(f_model, hash_bit=bit_list)
    model.set_train()
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=1e-4, weight_decay=1e-5)
    model = nn.TrainOneStepCell(model, optimizer)
    # Train the Model
    best_mAP = 0
    loss_meter = AverageMeter('loss')
    for epoch in range(args_opt.epoch):
        model.set_train(True)
        start = time.time()
        for i, data in enumerate(train_dataset.create_dict_iterator()):
            output = f_model(data["data"], data["label"]).asnumpy()
            loss = model(data["data"], data["label"])
            loss_meter.update(loss.asnumpy())

        time_used = (time.time() - start)
        per_step_time = time_used / dataset_size
        print('epoch: {}, step: {}, loss is {}, epoch time: {}s, per step time: {}s'.format(
            epoch, dataset_size, loss_meter, time_used, per_step_time))

        # Eval
        if epoch > 0 and epoch % args_opt.eval_step == 0:
            mAP5, mAP10, mAP15, mAP20 = eval(f_model, eval_dataset, train_dataset)
            if mAP20 >= best_mAP:
                best_mAP = mAP20
                mindspore.save_checkpoint(model, ckpt_save_path + "best.ckpt")
            print('mAP5: {}, mAP10: {}, mAP15 is {}, mAP20 is {}'.format(
                mAP5, mAP10, mAP15, mAP20))
            print(mAP5, mAP10, mAP15, mAP20)

        loss_meter.reset()
       

def eval(model, eval_dataset, train_dataset):
    model.set_train(False)
    bs_tst, clses_tst = [], []
    bs_trn, clses_trn = [], []
    for i, data in enumerate(eval_dataset.create_dict_iterator()):
        out = model(data["data"], label=None)
        bs_tst.append(out)
        clses_tst.append(data["label"])
    tst_binary = ops.cat(bs_tst).sign()
    tst_label = ops.cat(clses_tst)

    for i, data in enumerate(train_dataset.create_dict_iterator()):
        out = model(data["data"], label=None)
        bs_trn.append(out)
        clses_trn.append(data["label"])

    trn_binary = ops.cat(bs_trn).sign()
    trn_label = ops.cat(clses_trn)

    mAP5 = CalcTopMap(tst_binary.numpy(), trn_binary.numpy(), tst_label.numpy(), trn_label.numpy(), 5)
    mAP10 = CalcTopMap(tst_binary.numpy(), trn_binary.numpy(), tst_label.numpy(), trn_label.numpy(), 10)
    mAP15 = CalcTopMap(tst_binary.numpy(), trn_binary.numpy(), tst_label.numpy(), trn_label.numpy(), 15)
    mAP20 = CalcTopMap(tst_binary.numpy(), trn_binary.numpy(), tst_label.numpy(), trn_label.numpy(), 20)

    return mAP5, mAP10, mAP15, mAP20


def main():
    args = parse_args()
    set_seed(0)
    train(args)


if __name__ == '__main__':
    main()

