import torch
import pprint
import pandas as pd
import os.path as osp
from pretty_confusion_matrix import pp_matrix
from tqdm import tqdm
from networks import avail_models
from data import FreeMatchDataManager
import matplotlib.pyplot as plt
from utils import EMA, CELoss
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score
)

class FreeMatchTester:
    
    def __init__(
        self,
        cfg
    ):
        
        assert cfg.RESUME is not None and cfg.VALIDATE_ONLY
        
        self.cfg = cfg
        self.device = 'cuda' if cfg.USE_CUDA else 'cpu'
        self.ema_val = cfg.TRAINER.EMA_VAL
        self.ce_criterion = CELoss()
        
        if self.device == 'cuda':
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
        
        # Building model and setup EMA
        self.model = avail_models[cfg.MODEL.NAME](
            num_classes=cfg.DATASET.NUM_CLASSES,
            pretrained=cfg.MODEL.PRETRAINED,
            pretrained_path=cfg.MODEL.PRETRAINED_PATH
        )
        self.model = self.model.to(self.device)

        self.net = EMA(
            model=self.model,
            ema_decay=self.ema_val
        )
        self.net.train()
        
        # Build available dataloaders
        self.dm = FreeMatchDataManager(cfg.DATASET, cfg.TRAINER.NUM_TRAIN_ITERS)
        self.dm.data_statistics
        
        print('Loading model from the path: %s' % cfg.RESUME)
        self.__load__model__(cfg.RESUME)
        
    def __load__model__(self, load_path):

        ckpt = torch.load(load_path)
        self.net.model.load_state_dict(ckpt['model_state_dict'])
        self.net.load_state_dict(ckpt['ema_state_dict'])
        
        print('Model loaded from checkpoint. Path: %s' % load_path)
    
    @torch.no_grad()
    def test(self):

        self.net.eval()
        total_loss, total_num = 0, 0
        labels, preds = list(), list()
        for _, batch in enumerate(tqdm(self.dm.test_dl)):
            
            img_lb_w, label = batch['img_w'], batch['label']
            img_lb_w, label = img_lb_w.to(self.device), label.to(self.device)
            out = self.net(img_lb_w)
            logits = out['logits']
            loss = self.ce_criterion(logits, label, reduction='mean')
            labels.extend(label.cpu().tolist())
            preds.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            total_num += img_lb_w.shape[0]
            total_loss += loss.detach().item() * img_lb_w.shape[0]
           
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')       
        cm = confusion_matrix(labels, preds)
        cr = classification_report(labels, preds)

        test_report = {
            'loss': total_loss / total_num,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        save_dir = osp.join(self.cfg.LOG_DIR, self.cfg.RUN_NAME)
        file_name = open(osp.join(save_dir,'best_checkpoint_report.txt'), 'w')
        print('** Accuracy Report ** \n')
        pprint.pprint(test_report, indent=4)
        print('Other metrics are logged in the file: %s' % osp.join(save_dir,'best_checkpoint_report.txt'))
        df_cm = pd.DataFrame(cm, index=range(0, cm.shape[0]), columns=range(0, cm.shape[0]))
        pp_matrix(df_cm, cmap="PuRd", fz=8, cbar=True)
        plt.savefig(osp.join(save_dir, 'confusion_matrix.jpg'))
        plt.close()
        
        # Logging in the file
        print('Run Name: %s | Dataset: %s | Network Name: %s | Num labeled: %d \n' % (self.cfg.RUN_NAME, self.cfg.DATASET.NAME, self.cfg.MODEL.NAME, self.cfg.DATASET.NUM_LABELED), file=file_name)
        print(test_report, file=file_name)
        print('** Classification Report **\n', file=file_name)
        print(cr, file=file_name)
        print('** Confusion Matrix **\n', file=file_name)
        print(cm, file=file_name)
        file_name.close()