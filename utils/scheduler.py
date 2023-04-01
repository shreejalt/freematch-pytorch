import math
from torch.optim.lr_scheduler import LambdaLR

class FreeMatchScheduler:
    
    def __init__(
        self,
        optimizer,
        num_train_iters=1,
        num_cycles = 7. / 16
    ):
        self.num_train_iters = num_train_iters
        self.num_cycles = num_cycles
        self.num_warmup_iters = 0

        self.scheduler = LambdaLR(
            optimizer=optimizer.optimizer,
            lr_lambda=self.__lr__step__,
            last_epoch=-1
        )
    
    def step(self):
        
        self.scheduler.step()
    
    def __lr__step__(self, current_step):
        
        if current_step < self.num_warmup_iters:
            _lr = float(current_step) / float(max(1, self.num_warmup_iters))
        else:
            num_cos_steps = float(current_step - self.num_warmup_iters)
            num_cos_steps = num_cos_steps / float(max(1, self.num_train_iters - self.num_warmup_iters))
            _lr = max(0.0, math.cos(math.pi * self.num_cycles * num_cos_steps))
        return _lr