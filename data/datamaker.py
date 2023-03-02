from collections import namedtuple
import os
import os.path as osp
import random

DataTuple = namedtuple('DataTuple', ('impath', 'label'))

class DataMaker:
    
    def __init__(self, root, name, num_labeled=40):
        
        self.root = root
        self.name = name
        self.num_labeled = num_labeled
        
        train_dir = osp.join(root, name, 'train')
        test_dir = osp.join(root, name, 'test')
        ulb_dir = None
        if name == 'stl10':
            ulb_dir = osp.join(root, name, 'unlabeled')

        self.train_lb, self.train_ulb = self.__get_train_data__(
            train_dir=train_dir,
            num_labeled=num_labeled,
            ulb_dir=ulb_dir
        )
        
        self.test_lb = self.__get_test_data__(test_dir=test_dir)
        
    def __get_train_data__(self, train_dir, num_labeled, ulb_dir=None):
        
        classes = os.listdir(train_dir)
        classes.sort()
        imgs_per_class = num_labeled // len(classes)
        per_cls_cnt = 0
        train_lb, train_ulb = list(), list()
        
        for idx, cls in enumerate(classes):
            
            imnames = os.listdir(osp.join(train_dir, cls))
            random.shuffle(imnames)

            for imname in imnames:
                
                impath = osp.join(train_dir, cls, imname)
                if per_cls_cnt < imgs_per_class:
                    train_lb.append(DataTuple(impath, idx))
                    per_cls_cnt += 1
                else:
                    if ulb_dir is None:
                        train_ulb.append(DataTuple(impath, idx))
            per_cls_cnt = 0
        
        if ulb_dir is not None:
            train_ulb = self.__get_ulb_data__(ulb_dir=ulb_dir)
                
        return train_lb, train_ulb

    def __get_ulb_data__(self, ulb_dir):
        
        classes = os.listdir(ulb_dir)
        train_ulb = list()
        
        for _, cls in enumerate(classes):
            
            imnames = os.listdir(osp.join(ulb_dir, cls))
            
            for imname in imnames:
                impath = osp.join(ulb_dir, cls, imname)
                train_ulb.append(DataTuple(impath, -1))
            
        return train_ulb
        
    def __get_test_data__(self, test_dir):
        
        classes = os.listdir(test_dir)
        classes.sort()
        
        test_lb = list()
        
        for idx, cls in enumerate(classes):
            
            imnames = os.listdir(osp.join(test_dir, cls))
            random.shuffle(imnames)

            for imname in imnames:
                impath = osp.join(test_dir, cls, imname)
                test_lb.append(DataTuple(impath, idx))
        
        return test_lb
        
        
if __name__ == '__main__':
    
    root = 'data_download'
    name = 'cifar10'
    num_labeled = 40
    dm = DataMaker(root=root, name=name, num_labeled=num_labeled)
    print(len(dm.train_lb), len(dm.train_ulb), len(dm.test_lb))