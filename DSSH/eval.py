import os
import argparse
import yaml
import logging
import mindspore
from src.network import *
from src.data import *
from src.utils import *
from mindspore import context, load_checkpoint, load_param_into_net
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


def eval_net(args_opt):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    train_dataset = create_dataset(phase="train", batch_size=args_opt.train_batch_size,
                                   device_num=1,
                                   rank=0)
    eval_dataset = create_dataset(phase="eval", batch_size=args_opt.eval_batch_size,
                                  device_num=1,
                                  rank=0)
    print("eval dataset size: ", eval_dataset.get_dataset_size())
    ckpt_path = args_opt.output_dir + "/best.ckpt"
    model = Model(args_opt.bit_list)
    load_param_into_net(model, load_checkpoint(ckpt_path))
    mAP5, mAP10, mAP15, mAP20 = eval(model, eval_dataset, train_dataset)
    print('mAP5: {}, mAP10: {}, mAP15 is {}, mAP20 is {}'.format(
        mAP5, mAP10, mAP15, mAP20))


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


if __name__ == '__main__':
    args = parse_args()
    eval_net(args)
