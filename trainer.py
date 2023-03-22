import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os.path as osp
import os
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from data import FreeMatchDataManager
from networks import avail_models
import pprint
from utils import FreeMatchOptimizer, FreeMatchScheduler, TensorBoardLogger, EMA
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score
)

class CELoss:
    
    def __call__(self, logits, targets, reduction='none'):
        if logits.shape == targets.shape:
            preds = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * preds, dim=1)
            if reduction == 'none':
                return nll_loss
            return nll_loss.mean()
        else:
            preds = F.log_softmax(logits, dim=-1)
            return F.nll_loss(preds, targets, reduction=reduction)
'''      
class CELoss(nn.Module):
    
    def __init__(self):
        super(CELoss, self).__init__()
        
    def forward(self, logits, targets, reduction='none'):
        
        if logits.shape == targets.shape:
            preds = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * preds, dim=1)
            if reduction == 'none':
                return nll_loss
            return nll_loss.mean()
        else:
            preds = F.log_softmax(logits, dim=-1)
            return F.nll_loss(preds, targets, reduction=reduction)
'''
class ConsistencyLoss:
    
    def __call__(self, logits, targets, mask=None):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()
'''
class ConsistencyLoss(nn.Module):

    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
    def forward(self, logits, targets, mask=None):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()
'''
class SelfAdaptiveFairnessLoss:
    
    def __call__(self, mask, logits_ulb_s, p_t, label_hist):
        
        # Take high confidence examples based on Eq 7 of the paper
        logits_ulb_s = logits_ulb_s[mask.bool()]
        probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
        max_probs_s, max_idx_s = torch.max(probs_ulb_s, dim=-1)
        
        # Calculate the histogram of strong logits acc. to Eq. 9
        histogram = torch.bincount(max_idx_s, minlength=logits_ulb_s.shape[1]).to(logits_ulb_s.dtype)
        histogram /= histogram.sum()

        # Eq. 11 of the paper.
        p_t = p_t.reshape(1, -1)
        label_hist = label_hist.reshape(1, -1)
        
        scaler_p_t = self.__check__nans__(1 / label_hist).detach()
        modulate_p_t = p_t * scaler_p_t
        modulate_p_t /= modulate_p_t.sum(dim=-1, keepdim=True)
        
        scaler_prob_s = self.__check__nans__(1 / histogram).detach()
        modulate_prob_s = probs_ulb_s.mean(dim=0, keepdim=True) * scaler_prob_s
        modulate_prob_s /= modulate_prob_s.sum(dim=-1, keepdim=True)
        
        loss = (modulate_p_t * torch.log(modulate_prob_s + 1e-9)).sum(dim=1).mean()
        
        return loss, histogram.mean()

    @staticmethod
    def __check__nans__(x):
        x[x == float('inf')] = 0.0
        return x

'''
class SelfAdaptiveFairnessLoss(nn.Module):

    def __init__(self):
        super(SelfAdaptiveFairnessLoss, self).__init__()

    def forward(self, mask, logits_ulb_s, p_t, label_hist):
        
        # Take high confidence examples based on Eq 7 of the paper
        logits_ulb_s = logits_ulb_s[mask.bool()]
        probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
        max_probs_s, max_idx_s = torch.max(probs_ulb_s, dim=-1)
        
        # Calculate the histogram of strong logits acc. to Eq. 9
        histogram = torch.bincount(max_idx_s, minlength=logits_ulb_s.shape[1]).to(logits_ulb_s.dtype)
        histogram /= histogram.sum()

        # Eq. 11 of the paper.
        p_t = p_t.reshape(1, -1)
        label_hist = label_hist.reshape(1, -1)
        
        scaler_p_t = self.__check__nans__(1 / label_hist).detach()
        modulate_p_t = p_t * scaler_p_t
        modulate_p_t /= modulate_p_t.sum(dim=-1, keepdim=True)
        
        scaler_prob_s = self.__check__nans__(1 / histogram).detach()
        modulate_prob_s = probs_ulb_s.mean(dim=0, keepdim=True) * scaler_prob_s
        modulate_prob_s /= modulate_prob_s.sum(dim=-1, keepdim=True)
        
        loss = (modulate_p_t * torch.log(modulate_prob_s + 1e-9)).sum(dim=1).mean()
        
        return loss, histogram.mean()
        
    @staticmethod
    def __check__nans__(x):
        x[x == float('inf')] = 0.0
        return x
'''
class SelfAdaptiveThresholdLoss:
    def __init__(self, sat_ema):
        

        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()

    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())
        return tau_t, p_t, label_hist
    
    def __call__(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):

        tau_t, p_t, label_hist = self.__update__params__(logits_ulb_w, tau_t, p_t, label_hist)
        
        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)

        return loss, mask, tau_t, p_t, label_hist
'''
class SelfAdaptiveThresholdLoss(nn.Module):

    def __init__(self, sat_ema):
        
        super(SelfAdaptiveThresholdLoss, self).__init__()

        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()

    @torch.no_grad()
    def forward(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):

        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)

        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())

        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)

        return loss, mask, tau_t, p_t, label_hist
'''

class FreeMatchTrainer:

    def __init__(
            self,
            cfg
    ):
        
        self.cfg = cfg

        # Gathering the freematch training params.
        self.num_train_iters = cfg.TRAINER.NUM_TRAIN_ITERS
        self.num_eval_iters = cfg.TRAINER.NUM_EVAL_ITERS
        self.num_warmup_iters = cfg.TRAINER.NUM_WARMUP_ITERS
        self.num_log_iters = cfg.TRAINER.NUM_LOG_ITERS
        self.use_hard_labels = cfg.TRAINER.USE_HARD_LABELS
        self.ema_val = cfg.TRAINER.EMA_VAL
        self.soft_temp = cfg.TRAINER.SOFT_TEMP
        self.ulb_loss_ratio = cfg.TRAINER.ULB_LOSS_RATIO
        self.ent_loss_ratio = cfg.TRAINER.ENT_LOSS_RATIO
        self.device = 'cuda' if cfg.USE_CUDA else 'cpu'
        
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
        
        # Use Tensorboard if logging is enabled
        if cfg.USE_TB:
            self.tb = TensorBoardLogger(
                fpath=osp.join(cfg.LOG_DIR, cfg.RUN_NAME),
                filename=cfg.TB_DIR 
            )
        
        # Build available dataloaders
        self.dm = FreeMatchDataManager(cfg.DATASET, cfg.TRAINER.NUM_TRAIN_ITERS)
        self.dm.data_statistics

        # Build the optimizer and scheduler
        self.optim = FreeMatchOptimizer(self.model, cfg.OPTIMIZER)
        self.sched = FreeMatchScheduler(
            optimizer=self.optim,
            num_train_iters=self.num_train_iters,
            num_warmup_iters=self.num_warmup_iters
        )

        # Initializing the loss functions
        self.sat_criterion = SelfAdaptiveThresholdLoss(cfg.TRAINER.SAT_EMA)
        self.ce_criterion = CELoss()
        self.saf_criterion = SelfAdaptiveFairnessLoss()
        
        # Initialize the class params
        self.curr_iter = 0
        self.best_test_iter = -1
        self.best_test_acc = -1
        self.add_vis = cfg.PLOT_CURVES
        self.p_t = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        self.label_hist = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        self.tau_t = self.p_t.mean()

        self.amp = nullcontext
        if cfg.TRAINER.AMP_ENABLED:
            self.scaler = GradScaler()
            self.amp = autocast

        # Load Model if resume is true
        if cfg.CONT_TRAIN:
            print('Loading model from the path: %s' % cfg.RESUME)
            self.__load__model__(cfg.RESUME)

        self.__toggle__device__()
        
        
    def train(self):

        self.net.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        
        for (batch_lb, batch_ulb) in zip(self.dm.train_lb_dl, self.dm.train_ulb_dl):
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            img_lb_w, label_lb = batch_lb['img_w'], batch_lb['label']
            img_ulb_w, img_ulb_s = batch_ulb['img_w'], batch_ulb['img_s']
            
            img_lb_w, label_lb = img_lb_w.to(self.device), label_lb.to(self.device) 
            img_ulb_w, img_ulb_s = img_ulb_w.to(self.device), img_ulb_s.to(self.device)
            
            num_lb = img_lb_w.shape[0]
            num_ulb = img_ulb_w.shape[0]
            
            assert num_ulb == img_ulb_s.shape[0]
            
            img = torch.cat([img_lb_w, img_ulb_w, img_ulb_s])
            with self.amp():
                
                out = self.net(img)
                
                logits = out['logits']
                logits_lb = logits[:num_lb]
                logits_ulb_w, logits_ulb_s = logits[num_lb:].chunk(2)
                loss_lb = self.ce_criterion(logits_lb, label_lb, reduction='mean')
                loss_sat, mask, self.tau_t, self.p_t, self.label_hist = self.sat_criterion(
                    logits_ulb_w, logits_ulb_s, self.tau_t, self.p_t, self.label_hist
                )
                loss_saf, hist_p_ulb_s = self.saf_criterion(mask, logits_ulb_s, self.p_t, self.label_hist)                
                loss = loss_lb + self.ulb_loss_ratio * loss_sat + self.ent_loss_ratio * loss_saf
              
            if self.cfg.TRAINER.AMP_ENABLED:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()
            
            self.sched.step()
            self.net.update()
            self.optim.zero_grad()

            end_run.record()
            torch.cuda.synchronize()
            
            # Logging in tensorboard
            log_dict = {
                'train/lb_loss': loss_lb.item(),
                'train/sat_loss': loss_sat.item(),
                'train/saf_loss': loss_saf.item(),
                'train/total_loss': loss.item(),
                'train/mask': 1 - mask.mean().item(),
                'train/tau_t': self.tau_t.item(),
                'train/p_t': self.p_t.mean().item(),
                'train/label_hist': self.label_hist.mean().item(),
                'train/label_hist_s': hist_p_ulb_s.mean().item(),
                'train/lr': self.optim.optimizer.param_groups[0]['lr']
            } 
            
            if (self.curr_iter + 1) % self.num_eval_iters == 0:
                
                print('Evaluating...')
                validate_dict = self.validate()
                log_dict.update(validate_dict)
                save_dir = osp.join(self.cfg.LOG_DIR, self.cfg.RUN_NAME, self.cfg.OUTPUT_DIR)
                if not osp.exists(save_dir):
                    os.makedirs(save_dir)
                    
                if validate_dict['validation/accuracy'] > self.best_test_acc:
                    self.best_test_acc = validate_dict['validation/accuracy']
                    self.best_test_iter = self.curr_iter
                    self.__save__model__(save_dir, 'best_checkpoint.pth')
    
                self.__save__model__(save_dir, 'last_checkpoint.pth')
            
                log_dict.update(
                            {
                                'best_acc': self.best_test_acc,
                                'best_iter': self.best_test_iter
                            }
                )
                self.tb.update(log_dict, self.curr_iter)
                
            if (self.curr_iter + 1) % self.num_log_iters == 0:
                
                print('Iteration: %d / %d' % (self.curr_iter + 1, self.num_train_iters))
                print('Fetch Time: %.3f, Run Time: %.3f' % (start_batch.elapsed_time(end_batch) / 1000, start_run.elapsed_time(end_run) / 1000 ))
                pprint.pprint(log_dict, indent=4)

            self.curr_iter += 1
            del log_dict
            start_batch.record()

    @torch.no_grad()
    def validate(self):

        # self.model.eval()
        self.net.eval()
        total_loss, total_num = 0, 0
        labels, preds = list(), list()
        for _, batch in enumerate(self.dm.test_dl):
            
            img_lb_w, label = batch['img_w'], batch['label']
            img_lb_w, label = img_lb_w.to(self.device), label.to(self.device)
            out = self.net(img_lb_w)
            # out = self.model(img_lb_w)
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
        cf = confusion_matrix(labels, preds)
        cr = classification_report(labels, preds)

        print('Classification Report: \n')
        print(cr)
        
        print('Confusion Matrix \n')
        print(np.array_str(cf))

        self.net.train()
        # self.model.train()
        return {
            'validation/loss': total_loss / total_num,
            'validation/accuracy': acc,
            'validation/precision': precision,
            'validation/recall': recall,
            'validation/f1': f1
        }

    def __save__model__(self, save_dir, save_name='latest.ckpt'):

        save_dict = {
            'model_state_dict': self.net.model.state_dict(),
            'ema_state_dict':self.net.state_dict(),
            'optimizer_state_dict': self.optim.optimizer.state_dict(),
            'scheduler_state_dict': self.sched.scheduler.state_dict(),
            'curr_iter': self.curr_iter,
            'best_test_iter': self.best_test_iter,
            'best_test_acc': self.best_test_acc,
            'tau_t': self.tau_t.cpu(),
            'p_t': self.p_t.cpu(),
            'label_hist': self.p_t.cpu()
        }

        torch.save(save_dict, osp.join(save_dir, save_name))
        print('Model saved sucessfully. Path: %s' % osp.join(save_dir, save_name))


    def __load__model__(self, load_path):

        ckpt = torch.load(load_path)
        self.net.model.load_state_dict(ckpt['model_state_dict'])
        self.net.load_state_dict(ckpt['ema_state_dict'])
        self.optim.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.sched.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # Algorithm specfic loading
        self.curr_iter = ckpt['curr_iter']
        self.tau_t = ckpt['tau_t']
        self.p_t = ckpt['p_t']
        self.label_hist = ckpt['label_hist']
        self.best_test_iter = ckpt['best_test_iter']
        self.best_test_acc = ckpt['best_test_acc']
        
        
        print('Initialized checkpoint parameters..')
        print(f'Best Accuracy: {self.best_test_acc} Best Iteration: {self.best_test_iter}')
        print('Model loaded from checkpoint. Path: %s' % load_path)

    def __toggle__device__(self):
        
        self.p_t = self.p_t.to(self.device)
        self.tau_t = self.tau_t.to(self.device)
        self.label_hist = self.label_hist.to(self.device)