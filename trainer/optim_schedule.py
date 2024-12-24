'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim(): # 学习率预热，Learning Rate Warm-up
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, limit_lr, has_warmup=True):
        self._optimizer = optimizer
        self.limit_lr = limit_lr
        self.has_warmup = has_warmup
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5) # 初始学习率

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        if self.has_warmup:
            self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        # 一开始warm_up，其上升很快，到一定step后，采用np.power(self.n_current_steps, -0.5)，其比较稳定
        # 学习率预热，学习率小 --> 学习率增大 --> 学习率稳定
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])  

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        
        if self.n_current_steps > self.n_warmup_steps and lr <= self.limit_lr:
            lr = self.limit_lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
