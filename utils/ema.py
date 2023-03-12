import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


class EMA(nn.Module):

    def __init__(self, model, ema_decay=0.999):

        super(EMA, self).__init__()

        self.model = model
        self.decay = ema_decay
        self.ema = deepcopy(self.model)

        for _, param in self.ema.named_parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):

        if self.training:

            model_params = OrderedDict(self.model.named_parameters())
            ema_params = OrderedDict(self.model.named_parameters())

            assert model_params.keys() == ema_params.keys()

            for name, param in model_params.keys():
                if param.requires_grad:
                    ema_params[name] = self.decay * ema_params[name] + (1. - self.decay) * model_params[name]
            
            # Copy buffers if any like grad values to ema model
            model_buffers = OrderedDict(self.model.named_buffers())
            ema_buffers = OrderedDict(self.ema.named_buffers())

            assert model_buffers.keys() == ema_buffers.keys()

            for name, buffer in model_buffers.items():
                ema_buffers[name].copy_(buffer)
        else:
            raise AssertionError ('EMA can only be updated during training')
        
    def forward(self, x, fc_flag=False, feat_flag=False):
        if self.training:
            return self.model(x, fc_flag=fc_flag, feat_flag=feat_flag)
        else:
            return self.ema(x, fc_flag=fc_flag, feat_flag=feat_flag)
