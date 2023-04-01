from torchvision.datasets import SVHN, STL10, CIFAR10, CIFAR100
import numpy as np
import random


class DataMaker:
    
    def __init__(self, root, name, num_labeled=40):
        
        self.name = name
        self.root = root
        unlabeled_data = None
        if self.name == 'cifar10':
            train_data = CIFAR10(self.root, download=True, train=True)
            test_data = CIFAR10(self.root, train=False)
        elif self.name == 'cifar100':
            train_data = CIFAR100(self.root, download=True, train=True)
            test_data = CIFAR100(self.root, train=False)
        elif self.name == 'svhn':
            train_data = SVHN(self.root, split='train', download=True)
            test_data = SVHN(self.root, split='test', download=True)
        elif self.name == 'stl10':
            train_data = STL10(self.root, download=True, split='train')
            test_data = STL10(self.root, download=True, split='test')
            unlabeled_data = STL10(self.root, download=True, split='unlabeled')
        else:
            raise ValueError('Only CIFAR10, CIFAR100, SVHN, STL10 datasets are supported.') 
    
        self.train_lb, self.train_ulb = self.__get__train__data(
                train_data,
                num_labeled=num_labeled,
                ulb_data=unlabeled_data
            )
            
        self.test_lb = self.__get_test_data__(test_data=test_data)
        
        self.train_lb_cnt = self.__get_count__(self.train_lb['labels'])
        self.train_ulb_cnt = self.__get_count__(self.train_ulb['labels'])
        self.test_lb_cnt = self.__get_count__(self.test_lb['labels'])
    
    def __get__train__data(self, train_data, num_labeled, ulb_data=None):
        
        imgs, labels = train_data.data, train_data.targets
        imgs, labels = np.array(imgs), np.array(labels)

        classes = np.unique(labels)
        imgs_per_class = num_labeled // len(classes)
        train_lb, train_lb_labels = list(), list()
        train_ulb, train_ulb_labels = list(), list()
        
        for cls in classes:
            
            img_idxs = np.where(labels == cls)[0]
            labeled_idx = np.random.choice(img_idxs, imgs_per_class, False)
            
            train_lb.extend(imgs[labeled_idx])
            train_lb_labels.extend(labels[labeled_idx])
            
        train_ulb.extend(imgs)
        train_ulb_labels.extend(labels)
        
        if ulb_data is not None:
            train_ulb, train_ulb_labels = np.array(ulb_data.data).astype(np.uint8), np.ones(ulb_data.shape[0]) * -1.0

        return (
            {'images': np.array(train_lb).astype(np.uint8), 'labels': np.array(train_lb_labels)}, 
            {'images': np.array(train_ulb).astype(np.uint8), 'labels': np.array(train_ulb_labels)}
        )
    
    def __get_test_data__(self, test_data):
        
        imgs, labels = test_data.data, test_data.targets
        imgs, labels = np.array(imgs), np.array(labels)
        
        classes = np.unique(labels)
        test_lb, test_lb_labels = list(), list()
        
        for cls in classes:
            
            img_idxs = np.where(labels == cls)[0]
            random.shuffle(img_idxs)
            for idx in img_idxs:
                test_lb.append(imgs[idx])
                test_lb_labels.append(labels[idx])
        
        return  {'images': np.array(test_lb).astype(np.uint8), 'labels': np.array(test_lb_labels)}
        
    def __get_count__(self, data):
        
        val, cnt = np.unique(data, return_counts=True)
        return dict(zip(val, cnt))
    
if __name__ == '__main__':
    
    root = 'data_download'
    name = 'cifar10'
    num_labeled = 40
    dm = DataMaker(root=root, name=name, num_labeled=num_labeled)
    print(dm.train_lb['images'].shape, dm.train_ulb['images'].shape, dm.train_ulb['labels'].shape, dm.test_lb['labels'].shape)
    print(dm.train_lb_cnt, dm.train_ulb_cnt, dm.test_lb_cnt)