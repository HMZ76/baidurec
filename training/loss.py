import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomContrastiveLoss(nn.Module):
    def forward(self, logits, labels, pad_mask, ad_idxs):

        device = logits.device
        B, S, D = logits.shape
        logits_flat = logits.view(-1, D)
        labels_flat = labels.view(-1, D)
        mask_flat = pad_mask.view(-1)
        
        sim_matrix = torch.matmul(logits_flat, labels_flat.t())

        valid_mask = mask_flat.unsqueeze(0) * mask_flat.unsqueeze(1)
        sim_matrix = sim_matrix.masked_fill(valid_mask == 0, -1e9)
        
        probs = F.softmax(sim_matrix, dim=-1)
        pos_probs = torch.diag(probs)
        return -torch.log(pos_probs + 1e-8).mean()