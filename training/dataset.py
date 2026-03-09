import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from tqdm import tqdm

def read_ad_data(file_path, unitid_data):
    """解析 ad_data: ad_id \t features \t embedding"""
    print(f"正在读取广告数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            ad_id = int(parts[0])
            # parts[1] 是特征标签 (1,897,34...)，如果模型不需要可以不存
            # parts[2] 是 embedding (-0.0214233,-0.0187378...)
            embedding = np.fromstring(parts[2], sep=',', dtype=np.float32)
            unitid_data[ad_id] = {'embedding': embedding}

class BaseDataset(Dataset):
    def load_unit(self, root_dir):
        unitid_data = {}
        # 显式读取 ad_data 文件
        ad_data_path = os.path.join(root_dir, 'ad_data')
        if os.path.exists(ad_data_path):
            read_ad_data(ad_data_path, unitid_data)
        else:
            # 如果 ad_data 在子目录下
            for root, dirs, files in os.walk(root_dir):
                if 'ad_data' in files:
                    read_ad_data(os.path.join(root, 'ad_data'), unitid_data)
                    break
        
        return unitid_data, len(unitid_data)

class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.unitid_data, self.lenth_unit_data = self.load_unit(args.dataset_dir)
        self.train_data = []
        
        # sequence_data: user_id \t ad_id1 ad_id2 ...
        seq_data_path = os.path.join(args.dataset_dir, 'sequence_data')
        print(f"正在读取训练序列: {seq_data_path}")
        
        with open(seq_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                # parts[0] 是 user_id (hex string)，parts[1] 是空格分隔的 ad_ids
                raw_ad_ids = list(map(int, parts[1].split()))
                
                # 过滤掉不在 ad_data 中的广告 ID
                ad_filter_ids = [aid for aid in raw_ad_ids if aid in self.unitid_data]
                
                # 只有序列长度大于1才加入训练（因为要预测下一个）
                if len(ad_filter_ids) > 1:
                    self.train_data.append({'ad_ids': ad_filter_ids})

    def __len__(self): 
        return len(self.train_data)

    def __getitem__(self, idx):
        ad_ids = self.train_data[idx]['ad_ids']
        # 截断最大长度
        curr_ad_ids = ad_ids[-self.args.maxlen-1:] 
        ad_embeddings = [self.unitid_data[aid]['embedding'] for aid in curr_ad_ids]
        
        return ad_embeddings, curr_ad_ids

    def collate_fn(self, batch):
        ad_embeddings, ad_ids = zip(*batch)
        
        # 计算当前 batch 的最大长度 (需要 -1 因为输入比原始序列少一个)
        max_len = max(len(emb) - 1 for emb in ad_embeddings)
        emb_dim = self.args.emb_dim
        unit_data_keys = list(self.unitid_data.keys())

        padded_embeddings = []
        padded_pos_embs = []
        padded_neg_embs = []
        padded_ad_ids = []
        pad_mask = []

        for i in range(len(batch)):
            emb = ad_embeddings[i]
            ids = ad_ids[i]
            
            # 输入序列: [0 : N-1], 目标序列: [1 : N]
            input_seq = torch.tensor(np.array(emb[:-1]), dtype=torch.float32)
            pos_seq = torch.tensor(np.array(emb[1:]), dtype=torch.float32)
            pos_ids = torch.tensor(ids[1:], dtype=torch.long)
            
            curr_len = input_seq.shape[0]
            pad_len = max_len - curr_len
            
            # Padding (Pre-padding)
            zero_pad = torch.zeros([pad_len, emb_dim])
            id_pad = torch.zeros([pad_len], dtype=torch.long)
            
            padded_embeddings.append(torch.cat([zero_pad, input_seq], dim=0))
            padded_pos_embs.append(torch.cat([zero_pad, pos_seq], dim=0))
            padded_ad_ids.append(torch.cat([id_pad, pos_ids], dim=0))
            
            # Mask
            mask = torch.zeros(max_len)
            mask[pad_len:] = 1
            pad_mask.append(mask)
            
            # 负采样
            neg_ids = self.generate_random_ids(unit_data_keys, set(ids), curr_len)
            neg_emb_list = [self.unitid_data[nid]['embedding'] for nid in neg_ids]
            neg_seq = torch.tensor(np.array(neg_emb_list), dtype=torch.float32)
            padded_neg_embs.append(torch.cat([zero_pad, neg_seq], dim=0))

        return (torch.stack(padded_embeddings), 
                torch.stack(padded_pos_embs), 
                torch.stack(padded_neg_embs), 
                torch.stack(pad_mask), 
                torch.stack(padded_ad_ids))

    def generate_random_ids(self, all_keys, exclude_set, count):
        res = []
        while len(res) < count:
            rid = random.choice(all_keys)
            if rid not in exclude_set:
                res.append(rid)
        return res

class TestDataset(TrainDataset): # 继承以复用加载逻辑
    def __init__(self, args):
        self.args = args
        self.unitid_data, _ = self.load_unit(args.dataset_dir)
        self.test_data = []
        self.gt_data = []
        
        seq_data_path = os.path.join(args.dataset_dir, 'sequence_data')
        
        # 模拟测试：取每个序列的最后一个作为 GT，前面作为输入
        with open(seq_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                ad_ids = [int(x) for x in parts[1].split() if int(x) in self.unitid_data]
                
                if len(ad_ids) > 1:
                    self.test_data.append(ad_ids[:-1]) # 历史序列
                    self.gt_data.append(ad_ids[-1])    # 最后一个是预测目标

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        history_ids = self.test_data[idx][-self.args.maxlen:]
        history_embs = [self.unitid_data[aid]['embedding'] for aid in history_ids]
        gt_id = self.gt_data[idx]
        gt_emb = self.unitid_data[gt_id]['embedding']
        
        return history_embs, gt_id, gt_emb

    def collate_fn(self, batch):
        history_embs, gt_ids, gt_embs = zip(*batch)
        
        max_len = max(len(emb) for emb in history_embs)
        padded_embeddings = []
        pad_mask = []
        
        for emb in history_embs:
            emb_len = len(emb)
            pad_len = max_len - emb_len
            
            # 测试集通常用 0 填充而不是 torch.randn，保持推理一致性
            pad_vec = torch.zeros([pad_len, self.args.emb_dim])
            padded_emb = torch.cat([pad_vec, torch.tensor(np.array(emb))], dim=0)
            
            mask = torch.zeros(max_len)
            mask[pad_len:] = 1
            
            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)
            
        return (torch.stack(padded_embeddings), 
                torch.stack(pad_mask), 
                torch.tensor(gt_ids), 
                torch.tensor(np.array(gt_embs)))