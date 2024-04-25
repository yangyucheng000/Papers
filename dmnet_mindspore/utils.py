import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
CONFIG = {}


def set_global_config(cfg : dict):
    global CONFIG
    CONFIG = cfg

def change_global_config(name, content):
    global CONFIG
    CONFIG[name] = content


def get_global_config() -> dict:
    global CONFIG
    return CONFIG

def cal_acc(out: Tensor, labels: Tensor):
    softmax_scores = ops.softmax(out, axis=1)
    _, predict_labels = ops.min(softmax_scores, axis=1)
    pred = [1 if labels[i]==predict_labels[i] else 0 for i in range(labels.shape[0])]
    acc = np.mean(pred)
    return acc
