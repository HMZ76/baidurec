import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomContrastiveLoss(nn.Module):
    def __init__(self):
        super(CustomContrastiveLoss, self).__init__()

    def forward(self, logits, labels, pad_mask, ad_idxs):
        batch_size, seq_len, dim = logits.shape
        logits_flatten = logits.reshape(batch_size * seq_len, dim)
        labels_flatten = labels.reshape(batch_size * seq_len, dim)
        pad_mask = pad_mask.reshape(batch_size * seq_len)
        ad_idxs = ad_idxs.reshape(batch_size * seq_len)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(logits_flatten, labels_flatten.transpose(0, 1))
        # mask - 修复：确保所有张量在同一设备上
        device = logits.device
        mask = torch.zeros((batch_size * seq_len,batch_size * seq_len),dtype=torch.float32, device=device)
        mask = torch.where(pad_mask == 0, mask, torch.tensor(1.0, dtype=torch.float32, device=device)) # 纵行
        mask = torch.where(pad_mask.unsqueeze(-1).expand(-1, batch_size * seq_len) == 0, torch.tensor(0.0, dtype=torch.float32, device=device), torch.tensor(1.0, dtype=torch.float32, device=device)) # 横行
        similarity_matrix = similarity_matrix * mask
        sf = torch.nn.Softmax(dim=-1)
        similarity_matrix = sf(similarity_matrix)
        # loss
        label = (ad_idxs.unsqueeze(0) == ad_idxs.unsqueeze(-1))
        label = torch.where(label, torch.tensor(1.0, dtype=torch.float32, device=device), torch.tensor(0.0, dtype=torch.float32, device=device))
        label = torch.where(mask == 0, torch.tensor(0.0, dtype=torch.float32, device=device), label)
        loss = torch.where(label == torch.tensor(1.0, dtype=torch.float32, device=device), -torch.log2(similarity_matrix + 1e-8), torch.tensor(0.0, dtype=torch.float32, device=device))
        loss_sum = torch.sum(loss, axis=-1)
        # 返回平均损失
        return loss_sum.mean()
