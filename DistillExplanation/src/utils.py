import random
import numpy as np
import mindspore

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    mindspore.dataset.config.set_seed(seed)



from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

class LinearWithWarmUp(LearningRateSchedule):
    """
    Warmup-decay learning rate.
    """
    def __init__(self, learning_rate, num_warmup_steps, num_training_steps):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def construct(self, global_step):
        if global_step < self.num_warmup_steps:
            return global_step / float(max(1, self.num_warmup_steps)) * self.learning_rate
        return ops.maximum(
            0.0, (self.num_training_steps - global_step) / (max(1, self.num_training_steps - self.num_warmup_steps))
        ) * self.learning_rate