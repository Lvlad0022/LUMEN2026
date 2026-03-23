import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer

class LRWarmup:
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
    
    def __call__(self, optimizer, **kwargs):
        return self.scheduler(optimizer, **kwargs)
    
    def sheduler(self, optimizer):
        def scheduler_fn(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)
    
class WarmupStableDecayLR(lr_scheduler.LambdaLR):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            stable_steps: int,
            decay_steps: int,
            last_epoch: int = 1
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            elif step < warmup_steps + stable_steps:
                return 1.0
            else:
                steps_since_decay_start = step - warmup_steps - stable_steps
                return max(
                    0.0,
                    float(decay_steps - steps_since_decay_start) / float(decay_steps)

                )
            
        super(WarmupStableDecayLR, self).__init__(optimizer, lr_lambda, last_epoch) 