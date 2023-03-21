import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


class EMA(nn.Module):

    def __init__(self, model, device='cpu', ema_decay=0.999):

        super(EMA, self).__init__()

        self.model = model
        self.decay = ema_decay
        self.device = device
        self.ema = deepcopy(self.model)

        for _, param in self.ema.named_parameters():
            param.requires_grad_(False)
        
    @torch.no_grad()
    def update(self):

        if self.training:

            model_named_params = self.model.state_dict()
            
            for name, param in self.ema.state_dict().items():
                param_ = model_named_params[name].detach()
                param.copy_(param * self.decay + (1. - self.decay) * param_)

        else:
            raise AssertionError ('EMA can only be updated during training')
    
    def zero_grad(self):
        self.model.zero_grad()
        
    def forward(self, x, fc_flag=False, feat_flag=False):
        if self.training:
            return self.model(x, fc_flag=fc_flag, feat_flag=feat_flag)
        else:
            return self.ema(x, fc_flag=fc_flag, feat_flag=feat_flag)
