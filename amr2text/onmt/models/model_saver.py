import os
import pickle

import onmt.inputters

from collections import deque
from onmt.utils.logging import logger


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.save_checkpoint_steps,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def maybe_save(self, step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return

        chkpt, chkpt_name = self._save(step)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=0):
        super(ModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            save_checkpoint_steps, keep_checkpoint)

    def _save(self, step):
        real_model = self.model
        real_generator = real_model.generator

        model_state_dict = real_model.trainable_params()
        generator_state_dict = real_generator.trainable_params()
        save_list_model = [{"name": param.name, "data": param.data} for param in model_state_dict if "generator" not in param.name]
        save_list_generator = [{"name": param.name, "data": param.data} for param in generator_state_dict]
        checkpoint = {
            'model': save_list_model,
            'generator': save_list_generator,
            'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            # 'optim': self.optim,
        }

        logger.info("Saving checkpoint %s_step_%d.pkl" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pkl' % (self.base_path, step)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
