import torch.nn.functional as F
import torch

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

class ConsistencyLoss:
        
    def __call__(self, logits, targets, mask=None):
        
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets, reduction='none')
        if mask is not None:
            masked_loss = loss * mask.float()
            return masked_loss.mean()
        return loss.mean()

class SelfAdaptiveFairnessLoss:
    
    def __call__(self, mask, logits_ulb_s, p_t, label_hist):
        
        # Take high confidence examples based on Eq 7 of the paper
        logits_ulb_s = logits_ulb_s[mask.bool()]
        probs_ulb_s = torch.softmax(logits_ulb_s, dim=-1)
        max_idx_s = torch.argmax(probs_ulb_s, dim=-1)
        
        # Calculate the histogram of strong logits acc. to Eq. 9
        # Cast it to the dtype of the strong logits to remove the error of division of float by long
        histogram = torch.bincount(max_idx_s, minlength=logits_ulb_s.shape[1]).to(logits_ulb_s.dtype)
        histogram /= histogram.sum()

        # Eq. 11 of the paper.
        p_t = p_t.reshape(1, -1)
        label_hist = label_hist.reshape(1, -1)
        
        # Divide by the Sum Norm for both the weak and strong augmentations
        scaler_p_t = self.__check__nans__(1 / label_hist).detach()
        modulate_p_t = p_t * scaler_p_t
        modulate_p_t /= modulate_p_t.sum(dim=-1, keepdim=True)
        
        scaler_prob_s = self.__check__nans__(1 / histogram).detach()
        modulate_prob_s = probs_ulb_s.mean(dim=0, keepdim=True) * scaler_prob_s
        modulate_prob_s /= modulate_prob_s.sum(dim=-1, keepdim=True)
        
        # Cross entropy loss between two Sum Norm logits. 
        loss = (modulate_p_t * torch.log(modulate_prob_s + 1e-9)).sum(dim=1).mean()
        
        return loss, histogram.mean()

    @staticmethod
    def __check__nans__(x):
        x[x == float('inf')] = 0.0
        return x

class SelfAdaptiveThresholdLoss:
    
    def __init__(self, sat_ema):
        
        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()
        
    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        
        # Updating the histogram for the SAF loss here so that I dont have to call the torch.no_grad() function again. 
        # You can do it in the SAF loss also, but without accumulating the gradient through the weak augmented logits
        
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

