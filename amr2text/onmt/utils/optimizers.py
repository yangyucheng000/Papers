""" Optimizers class """
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import mindspore
from onmt.utils import use_gpu


def build_optim(model, opt, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
        params=model.get_parameters(),
        learning_rate=opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_steps=opt.start_decay_steps,
        decay_steps=opt.decay_steps,
    )
    return optim


class Optimizer(nn.SGD):
    """定义优化器"""
    def __init__(self, params, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 ):
        super().__init__(params, learning_rate)
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.start_decay_steps = Parameter(Tensor([start_decay_steps], mindspore.float32), name="start_decay_steps")
        self.decay_steps = decay_steps
        self.start_decay = Parameter(Tensor([0], mindspore.int32))
        self._step = Parameter(Tensor([0], mindspore.float32), name="step")
        self.assign_step = ops.Assign()
        self.assign_lr = ops.Assign()
        self.assign_start_decay = ops.Assign()

    def construct(self, gradients):
        # 待更新的权重参数
        self.assign_step(self._step, self._step + 1)
        lr = self.get_lr()
        if ((self.start_decay_steps is not None) and (
                self._step >= self.start_decay_steps)):
            self.assign_start_decay(self.start_decay, 1)
        if self.start_decay == Tensor([1], dtype=mindspore.int32):
            if ((self._step - self.start_decay_steps)
                    % self.decay_steps == 0):
                lr = lr * self.lr_decay
                self.assign_lr(self.learning_rate, lr)
        if self.max_grad_norm:
            gradients = ops.clip_by_global_norm(gradients, self.max_grad_norm)
        success = super().construct(gradients)
        return success



# class Optimizer(nn.SGD):
#     """定义优化器"""
#     def __init__(self, params, learning_rate, max_grad_norm,
#                  lr_decay=1, start_decay_steps=None, decay_steps=None,
#                  ):
#         print("type(params): ", type(params))
#         super(Optimizer, self).__init__(params, learning_rate)
#         self.last_ppl = None
#         self.max_grad_norm = max_grad_norm
#         self.lr_decay = lr_decay
#         self.start_decay_steps = Parameter(Tensor([start_decay_steps], mindspore.float32), name="step")
#         self.decay_steps = decay_steps
#         self.start_decay = False
#         self._step = Parameter(Tensor([1], mindspore.float32), name="step")
#         self.assign = ops.Assign()
#
#     def construct(self, gradients):
#         # 待更新的权重参数
#         self._step += 1
#         lr = self.get_lr()
#         if ((self.start_decay_steps is not None) and (
#                 self._step >= self.start_decay_steps)):
#             self.start_decay = True
#         if self.start_decay:
#             if ((self._step - self.start_decay_steps)
#                     % self.decay_steps == 0):
#                 lr = lr * self.lr_decay
#                 self.assign(self.learning_rate, lr)
#         if self.max_grad_norm:
#             self.parameters = ops.clip_by_global_norm(self.parameters, self.max_grad_norm)
#         success = super(Optimizer, self).construct(gradients)
#         return success


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])