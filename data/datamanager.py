from data import DataMaker, MyDataset, RandAugment
from torchvision import transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tabulate import tabulate
from torch.utils.data.sampler import BatchSampler, RandomSampler

class FreeMatchDataManager:
    
    def __init__(
        self,
        cfg,
        num_train_iters
    ):
        self.cfg = cfg
        self.num_train_iters = num_train_iters
           
        # Gather labeled, unlabeled, and test data
        self.datamaker = DataMaker(root=cfg.DIR, name=cfg.NAME, num_labeled=cfg.NUM_LABELED)
        self.train_lb_cnt, self.train_ulb_cnt = self.datamaker.train_lb_cnt, self.datamaker.train_ulb_cnt
        self.test_lb_cnt = self.datamaker.test_lb_cnt
        
        # Calculate the histogram of the data 
        self.data_dist = self.__get_data_dist__(self.datamaker.train_lb)
        
        train_weak_tfm, train_strong_tfm, test_tfm = self.__get_transforms__(self.cfg)
        
        train_lb_data = MyDataset(
            data=self.datamaker.train_lb,
            train_weak_tfm=train_weak_tfm,
            num_classes=cfg.NUM_CLASSES,
            train=True,
            convert_one_hot=cfg.CONVERT_ONE_HOT
        )
        train_ulb_data = MyDataset(
            data=self.datamaker.train_ulb,
            num_classes=cfg.NUM_CLASSES,
            train=True,
            train_weak_tfm=train_weak_tfm,
            train_strong_tfm=train_strong_tfm,
            convert_one_hot=cfg.CONVERT_ONE_HOT
        )
        
        test_data = MyDataset(
            data=self.datamaker.test_lb,
            test_tfm=test_tfm,
            train=False,
            convert_one_hot=cfg.CONVERT_ONE_HOT    
        )
        
        self.train_lb_dl = self.__get_dataloader__(train_lb_data, cfg.TRAIN_BATCH_SIZE, cfg.NUM_WORKERS, num_iters=self.num_train_iters)
        self.train_ulb_dl = self.__get_dataloader__(train_ulb_data, cfg.TRAIN_BATCH_SIZE * cfg.URATIO, cfg.NUM_WORKERS, num_iters=self.num_train_iters)
        self.test_dl = self.__get_dataloader__(test_data, cfg.TEST_BATCH_SIZE, cfg.NUM_WORKERS, train=False)

    @staticmethod 
    def __get_data_dist__(data_lb):
        
        _, cnt = np.unique(data_lb['labels'], return_counts=True)
        dist = cnt / cnt.sum()
        
        return dist
        
    @property
    def data_statistics(self):
        
        print('Data Statictics ...')

        if self.cfg.NAME != 'svhn':
            headers = ['Class', 'train labeled', 'train unlabeled', 'test labeled']
            table = [['%d' % cls, '%d' % self.train_lb_cnt[cls], '%d' % self.train_ulb_cnt[cls], '%d' % self.test_lb_cnt[cls]] for cls in self.train_lb_cnt.keys()]
        else:
            headers = ['Class', 'train labeled', 'test labeled']
            table = [['%d' % cls, '%d' % self.train_lb_cnt[cls], '%d' % self.test_lb_cnt[cls]] for cls in self.train_lb_cnt.keys()]
            print('Unlabeled data: %d', self.train_ulb_cnt[-1])
            
        print(tabulate(table, headers=headers))
            
    @staticmethod
    def __get_dataloader__(data, batch_size, num_workers, num_iters=1, train=True):
        
        if not train:
            return DataLoader(
                data,
                batch_size=batch_size,
                num_workers=num_workers
            )
        
        sampler = RandomSampler(data, replacement=True, num_samples=num_iters * batch_size, generator=None) 
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        return DataLoader(
            data,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )
        
    @staticmethod
    def __get_transforms__(cfg):
        
        # Adapting the policy from UDA(2020): https://arxiv.org/abs/1904.12848
        
        train_weak_tfm = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                T.ToTensor(),
                T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        train_strong_tfm = T.Compose(
           [
                RandAugment(cfg.RANDAUG_N),
                T.RandomHorizontalFlip(),
                T.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                T.ToTensor(),
                T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        test_tfm = T.Compose(
            [
                T.Resize([cfg.IMG_SIZE, cfg.IMG_SIZE]),
                T.ToTensor(),
                T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        return train_weak_tfm, train_strong_tfm, test_tfm