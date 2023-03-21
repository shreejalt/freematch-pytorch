from collections import namedtuple
import os.path as osp
import os
from torchvision.datasets import SVHN, STL10, CIFAR10, CIFAR100
from tqdm import tqdm

def mkdir(dir):
    
    if not osp.exists(dir):
        os.makedirs(dir)

class PrepareDataset:
    
    def __init__(self, name='cifar10', root='./data'):
        
        self.root = root
        self.name = name
    
    def prepare_data(self):

        self.__get_data__()
        
    def __get_data__(self):
        
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
            unlabeled_dir = osp.join(self.root, self.name, 'unlabeled')
            mkdir(unlabeled_dir)
        else:
            raise ValueError('Only CIFAR10, CIFAR100, SVHN, STL10 datasets are supported.')    
        
        train_dir = osp.join(self.root, self.name, 'train')
        test_dir = osp.join(self.root, self.name, 'test')
        mkdir(train_dir)
        mkdir(test_dir)
        
        print('Preparing dataset: %s' % self.name)
        self.__format_data__(train_data, train_dir)
        self.__format_data__(test_data, test_dir)
        if self.name == 'stl10':
            self.__format_data__(unlabeled_data, unlabeled_dir)
        
        
    @staticmethod
    def __format_data__(data, dir):
        for idx, d in tqdm(enumerate(data)):
            img, cls = d
            clspath = osp.join(dir, str(cls).zfill(3))
            mkdir(clspath)
            imgpath = osp.join(clspath, str(idx + 1).zfill(5) + '.jpg')
            img.save(imgpath)
    

if __name__ == '__main__':
    
    
    prepare = PrepareDataset(name='cifar10', root='./data_download')
    prepare.prepare_data()
    