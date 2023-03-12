import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from data import FreeMatchDataManager
from networks import avail_models
from utils import FreeMatchOptimizer, FreeMatchScheduler, TensorBoardLogger, EMA


class ConsistencyLoss(nn.Module):

    def forward(logits, targets, mask=None):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()

class SelfAdaptiveFairnessLoss(nn.Module):

    def __init__(self):
        super(SelfAdaptiveFairnessLoss, self).__init__()

    def forward(mask, logits_ulb_s, p_t, label_hist):
        pass



class SelfAdaptiveThresholdLoss(nn.Module):

    def __init__(self, sat_ema):
        
        super(SelfAdaptiveThresholdLoss, self).__init__()

        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()

    def forward(self, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):

        probs_ulb_w = torch.softmax(logits_ulb_w).detach()
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1, keepdim=True)

        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())

        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = max_probs_w.ge(tau_t * tau_t_c[max_idx_w]).to(max_probs_w.dtype)

        loss = self.criterion(logits_ulb_s, max_idx_w, mask=mask)

        return loss, mask

class FreeMatchTrainer:

    def __init__(
            self,
            cfg
    ):
        
        self.cfg = cfg

        # Gathering the freematch training params.
        self.num_train_iters = cfg.TRAINER.NUM_TRAIN_ITERS
        self.num_eval_iters = cfg.TRAINER.NgUM_EVAL_ITERS
        self.num_warmup_iters = cfg.TRAINER.NUM_WARMUP_ITERS
        self.num_log_iters = cfg.TRAINER.NUM_LOG_ITERS
        self.use_hard_labels = cfg.TRAINER.USE_HARD_LABELS
        self.ema_val = cfg.TRAINER.EMA_VAL
        self.soft_temp = cfg.TRAINER.SOFT_TEMP
        self.ulb_loss_ratio = cfg.TRAINER.ULB_LOSS_RATIO
        self.ent_loss_ratio = cfg.TRAINER.ENT_LOSS_RATIO

        # Building model and setup EMA
        model = avail_models[cfg.MODEL.NAME](
            num_classes=cfg.DATASET.NUM_CLASSES,
            pretrained=cfg.MODEL.PRETRAINED,
            pretrained_path=cfg.MODEL.PRETRAINED_PATH
        )

        self.model = EMA(
            model=model,
            ema_decay=self.ema_val
        )

        # Use Tensorboard if logging is enabled
        if cfg.USE_TB:
            self.tb = TensorBoardLogger(
                fpath=cfg.OUTPUT_DIR,
                filename=cfg.TB_FILENAME
            )
        
        # Build available dataloaders
        dm = FreeMatchDataManager(cfg.DATASET)
        dm.data_statistics

        # Build the optimizer and scheduler
        self.optim = FreeMatchOptimizer(self.model, cfg.OPTIMIZER)
        self.sched = FreeMatchScheduler(
            optimizer=self.optim,
            num_train_iters=self.num_train_iters,
            num_warmup_iters=self.num_warmup_iters
        )

        # Initialize the class params
        self.curr_iter = 0
        self.best_test_iter = -1
        self.best_test_acc = -1
        self.add_vis = cfg.PLOT_CURVES
        self.p_t = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        self.label_hist = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        self.tau_t = self.p_t.mean()

        amp = nullcontext
        if cfg.TRAINER.AMP_ENABLED:
            scaler = GradScaler()
            amp = autocast

        # Load Model if resume is true
        if cfg.CONT_TRAIN:
            print('Loading model from the path: %s' % cfg.RESUME)
            self.__load__model__(cfg.RESUME)

    def __save__model__(self, save_dir, save_name='latest.ckpt'):

        save_dict = {
            'model_state_dict': self.model.model.state_dict(),
            'ema_state_dict':self.model.ema.state_dict(),
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
        self.model.model.load_state_dict(ckpt['model_state_dict'])
        self.model.ema.load_state_dict(ckpt['ema_state_dict'])
        self.optim.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.sched.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # Algorithm specfic loading
        self.curr_iter = ckpt['curr_iter']
        self.tau_t = ckpt['tau_t']
        self.p_t = ckpt['p_t']
        self.label_hist = ckpt['label_hist']
        self.best_iter = ckpt['best_test_iter']
        self.best_acc = ckpt['best_test_acc']

        print('Model loaded from checkpoint. Path: %s' % load_path)
