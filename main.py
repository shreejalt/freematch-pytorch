from configs import CFG_default
from data import FreeMatchDataManager
from networks import wrn_28_2
from utils import FreeMatchOptimizer



if __name__ == '__main__':
    
    dm = FreeMatchDataManager(CFG_default.DATASET)
    print(dm.data_statistics)
    model = wrn_28_2(num_classes=10, pretrained=False)
    print(model)
    optim = FreeMatchOptimizer(model, CFG_default.OPTIMIZER)
    print(optim)