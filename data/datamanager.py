from data import DataMaker, PrepareDataset, MyDataset, RandAugment
from torchvision import transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tabulate import tabulate


class InfiniteDataLoader(DataLoader): 
    '''Infitely loading the batch from the dataset
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dset_iter = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dset_iter)
        except StopIteration:
            self.dset_iter = super().__iter__()
            batch = next(self.dset_iter)
        return batch

class FreeMatchDataManager:
    
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
        self.datamaker = DataMaker(root=cfg.DIR, name=cfg.NAME, num_labeled=cfg.NUM_LABELED)
        self.train_lb_cnt, self.train_ulb_cnt = self.datamaker.train_lb_cnt, self.datamaker.train_ulb_cnt
        self.test_lb_cnt = self.datamaker.test_lb_cnt
        
        # Calculate the histogram of the data 
        self.data_dist = self.__get_data_dist__(self.datamaker.train_lb, cfg.NUM_CLASSES)
        
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
        
        self.train_lb_dl = self.__get_dataloader__(train_lb_data, cfg.TRAIN_BATCH_SIZE, cfg.NUM_WORKERS)
        self.train_ulb_dl = self.__get_dataloader__(train_ulb_data, cfg.TRAIN_BATCH_SIZE * cfg.URATIO, cfg.NUM_WORKERS)
        self.test_dl = self.__get_dataloader__(test_data, cfg.TEST_BATCH_SIZE, cfg.NUM_WORKERS, shuffle=False, train=False)

    @staticmethod 
    def __get_data_dist__(data_lb, num_classes):
        
        dist = np.zeros((num_classes))
        for dt in data_lb:
            dist[dt.label] += 1
        dist /= dist.sum()
        
        return dist
        
    @property
    def data_statistics(self):
        
        print('Data Statictics ...')

        headers = ['Class', 'train labeled', 'test_labeled']
        table = [['%d' % cls, '%d' % self.train_lb_cnt[cls], '%d' % self.test_lb_cnt[cls]] for cls in self.train_lb_cnt.keys()]
        
        print(tabulate(table, headers=headers))
        
        print('Unlabeled data: %d' % self.train_ulb_cnt[-1])
    
    @staticmethod
    def __get_dataloader__(data, batch_size, num_workers, shuffle=True, train=True):
        
        loader = DataLoader if not train else InfiniteDataLoader
        
        return loader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        
    @staticmethod
    def __get_transforms__(cfg):
        
        # Adapting the policy from UDA(2020): https://arxiv.org/abs/1904.12848
        
        train_weak_tfm = T.Compose(
            [
                T.Resize(cfg.IMG_SIZE),
                T.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(cfg.PIXEL_MEAN, cfg.PIXEL_STD)
            ]
        )
        
        train_strong_tfm = T.Compose(
           [
                T.Resize(cfg.IMG_SIZE),
                T.RandomCrop([cfg.IMG_SIZE, cfg.IMG_SIZE], padding=int(cfg.IMG_SIZE * (1 - cfg.CROP_RATIO)), padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                RandAugment(cfg.RANDAUG_M, cfg.RANDAUG_N),
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