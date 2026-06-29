from torch.optim.lr_scheduler import _LRScheduler
import math

class NoOpScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoOpScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, epoch=None):
        pass

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, T_max, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.step_count = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            lr = (self.max_lr / self.warmup_steps) * (self.step_count + 1)
        else:
            lr = self.eta_min + 0.5 * (self.max_lr - self.eta_min) * (
                1 + math.cos(math.pi * (self.step_count - self.warmup_steps) / self.T_max)
            )
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.step_count += 1
        super().step(epoch)
