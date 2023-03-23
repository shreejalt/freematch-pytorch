from torch.optim import SGD, AdamW

class FreeMatchOptimizer:
    
    def __init__(
        self,
        model,
        cfg
    ):
        self.name = cfg.NAME
        self.lr = cfg.LR
        self.momentum = cfg.MOMENTUM
        self.weight_decay = cfg.WEIGHT_DECAY
        self.skip_bn = cfg.SKIP_BN
        self.nesterov = cfg.NESTEROV
        wd_params = self.__get__wd__params__(model=model, skip_bn=self.skip_bn)
        self.optimizer = self.__get__optimizer__(
            name=self.name,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            wd_params=wd_params,
            nesterov=self.nesterov
        ) 
    
    def step(self):    
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def __repr__(self):

        return (f'Name: {self.name} lr: {self.lr} momentum: {self.momentum} weight decay: {self.weight_decay}')
    
    @staticmethod
    def __get__wd__params__(model, skip_bn):
        
        decay_params, no_decay_params = list(), list()
        
        if skip_bn:
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if ('bn' in name or 'bias' in name) and param.ndim <= 1:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        return {
            'decay_params': decay_params,
            'no_decay_params': no_decay_params
        }
    
    @staticmethod
    def __get__optimizer__(name, lr, momentum, weight_decay, wd_params, nesterov):
        
        param_args = [
            {'params': wd_params['decay_params']},
            {'params': wd_params['no_decay_params'], 'weight_decay': 0.0}
        ]
        
        if name == 'SGD':
            return SGD(
                param_args,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )
        elif name == 'AdamW':
            return AdamW(
                param_args,
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError ('Only SGD and AdamW is supported')
        