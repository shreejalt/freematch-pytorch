from data import DataMaker, PrepareDataset, MyDataset
from torchvision import transforms
from transforms import RandAugment
import numpy as np
from torch.utils.data import DataLoader

class DataManager:
    
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
       
        # Download and format the data if not
        if cfg.PREPARE:
           prepare = PrepareDataset(name=cfg.NAME, root=cfg.DIR)
           prepare.prepare_data()
           
        # Gather labeled, unlabeled, and test data
        datamaker = DataMaker(root=cfg.DIR, name=cfg.NAME, num_labeled=cfg.NUM_LABELED)
        
        # Calculate the histogram of the data 
        self.data_dist = self.__get_data_dist__(datamaker.train_lb, cfg.NUM_CLASSES)
        
        train_weak_tfm, train_strong_tfm, test_tfm = self.__get_transforms__(self.cfg)
        
        train_lb_data = MyDataset(
            data=datamaker.train_lb,
            num_classes=cfg.NUM_CLASSES,
            train=True,
            convert_one_hot=cfg.CONVERT_ONE_HOT
        )
        
        train_ulb_data = MyDataset(
            data=datamaker.train_ulb,
            num_classes=cfg.NUM_CLASSES,
            train=True,
            train_weak_tfm=train_weak_tfm,
            train_strong_tfm=train_strong_tfm,
            convert_one_hot=cfg.CONVERT_ONE_HOT
        )
        
        test_data = MyDataset(
            data=datamaker.test_lb,
            test_tfm=test_tfm,
            train=False,
            convert_one_hot=cfg.CONVERT_ONE_HOT    
        )
       
        
    def __get_data_dist__(data_lb, num_classes):
        
        dist = np.zeros((num_classes))
        for dt in data_lb:
            dist[dt['label']] += 1
        dist /= dist.sum()
        
        return dist.tolist()
        
    @staticmethod
    def __get_dataloader__():
        pass
        
        
    @staticmethod
    def __get_transforms__(cfg):
        
        # Adapting the policy from UDA(2020): https://arxiv.org/abs/1904.12848
        
        train_weak_tfm = transforms.Compose(
            [
                transforms.Resize(cfg.IMG_SIZE),
                transforms.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        train_strong_tfm = transforms.Compose(
           [
                transforms.Resize(cfg.IMG_SIZE),
                transforms.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                RandAugment(cfg.RANDAUG_M, cfg.RANDAUG_N),
                transforms.ToTensor(),
                transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        test_tfm = transforms.Compose(
            [
                transforms.Resize([cfg.IMG_SIZE, cfg.IMG_SIZE]),
                transforms.ToTensor(),
                transforms.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        return train_weak_tfm, train_strong_tfm, test_tfm
        
                

