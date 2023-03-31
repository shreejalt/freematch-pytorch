from collections import OrderedDict

class EMA:
    
    def __init__(self, model, ema_decay=0.999):
        
        self.model = model
        self.ema_decay = ema_decay
        self.ema = self.__register__()        
        self.backup = dict()
        self.training = True
    
    def state_dict(self):
        
        return OrderedDict(self.ema.items())
    
    def load_state_dict(self, state):
        
        for key, value in state.items():
            assert key in self.ema.keys()
            self.ema[key] = value
        
    def train(self):
        
        if not self.training:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name]
    
            self.backup = dict()
            self.training = True
            
        self.model.train()
        
    def eval(self):
        
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema
                self.backup[name] = param.data
                param.data = self.ema[name]

        self.training = False
    
    def update(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema
                update = self.ema_decay * self.ema[name] + (1.0 - self.ema_decay) * param.data
                self.ema[name] = update.clone()
    
    def __register__(self):
        
        self.model.train()
        ema = dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ema[name] = param.data.clone()

        return ema
    
    def __call__(self, x, feat_flag=None, fc_flag=None):
        
        return self.model(x, feat_flag, fc_flag)

'''
class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def state_dict(self):
        
        return OrderedDict(self.shadow.items())
    
    def load_state_dict(self, state):
        
        for key, value in state.items():
            assert key in self.shadow.keys()
            self.shadow[key] = value

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
'''