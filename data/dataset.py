import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    
    def __init__(
        self, 
        data,
        num_classes=10,
        train_weak_tfm=None,
        train_strong_tfm=None,
        test_tfm=None,
        train=False,
        convert_one_hot=False,
    ):
        
        assert train_weak_tfm is not None
        
        self.train = train
        self.data = data
        
        self.num_classes = num_classes
        
        self.train_weak_tfm = train_weak_tfm
        self.train_strong_tfm = train_strong_tfm
        self.test_tfm = test_tfm
        
        self.convert_one_hot = convert_one_hot
                
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        impath, label = self.data[idx]
        
        img = Image.open(impath).convert("RGB")
        img_s = torch.empty((img.size), dtype=img.dtype)
        
        if self.train:
            img_w = self.train_weak_tfm(img)
            
            if self.train_strong_tfm is not None:
                img_s = self.train_strong_tfm(img)    
        else:
            img_w = self.test_tfm(img)
        
        if self.convert_one_hot:
            label = self.__convert_one_hot__(label, self.num_classes)
            
        return {
            'img_w': img_w,
            'label': label,
            'img_s': img_s
        }
        
    @staticmethod
    def __convert_one_hot__(label, num_classes):
        
        label_ = np.zeros(num_classes)
        label_[label] = 1
        return label_
        