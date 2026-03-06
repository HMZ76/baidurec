import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from tqdm import tqdm

def read_data(f, unitid_data):
    with open(f, 'r') as file:
        for line in tqdm(file):
            parts = line.strip().split('\t')
            ad_id = int(parts[0])
            embedding = list(map(np.float32, parts[2].split(',')))
            unitid_data[ad_id] = {'embedding': embedding}

def safe_process_file(f, unitid_data):
    try:
        read_data(f, unitid_data)
    except Exception as e:
        print(f"Error processing {f}: {e}")

class BaseDataset(Dataset):
    """提取通用的加载 unit 逻辑"""
    def load_unit(self, root_dir):
        print(f"开始加载unit数据: {root_dir}")
        file_list = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files]
        unitid_data = {}
        for f in file_list:
            safe_process_file(f, unitid_data)
        return unitid_data, len(unitid_data)

class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        
        data_root = os.path.join(args.dataset_dir, "1w_tokenized_unitid")
        self.unitid_data, self.lenth_unit_data = self.load_unit(data_root)
        self.train_data = []
        
        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split()))[:-1]
                ad_filter_ids = [aid for aid in ad_ids if aid in self.unitid_data]
                self.train_data.append({'ad_ids': ad_filter_ids})

    def __len__(self): return len(self.train_data)

    def __getitem__(self, idx):
        ad_ids = self.train_data[idx]['ad_ids']
        ad_embeddings = [self.unitid_data[aid]['embedding'] for aid in ad_ids]
        return ad_embeddings[-self.args.maxlen:], ad_ids[-self.args.maxlen:]

            
    def collate_fn(self, batch):
        # ad_embeddings: List of [seq_len, emb_dim]
        # ad_ids: List of [seq_len]
        ad_embeddings, ad_ids = zip(*batch)
        

        max_len = max(len(emb) - 1 for emb in ad_embeddings)
        emb_dim = self.args.emb_dim
        
        padded_embeddings = []
        padded_pos_embs = []
        padded_neg_embs = []
        pad_mask = []
        padded_ad_ids = []
        
        # 提前获取 unitid_data 的 key 列表，用于索引采样
        unit_data_keys = list(self.unitid_data.keys())

        for idx, emb in enumerate(ad_embeddings):
            # 原始序列去掉最后一个，作为输入序列
            input_seq = torch.tensor(emb[:-1], dtype=torch.float32)
            # 原始序列去掉第一个，作为正样本序列
            pos_seq = torch.tensor(emb[1:], dtype=torch.float32)
            # 对应的正样本 ID
            pos_ids = torch.tensor(ad_ids[idx][1:], dtype=torch.long)
            
            current_seq_len = input_seq.shape[0]
            padding_len = max_len - current_seq_len
            

            zero_padding = torch.zeros([padding_len, emb_dim])
            id_padding = torch.zeros([padding_len], dtype=torch.long)
            
            padded_embeddings.append(torch.cat([zero_padding, input_seq], dim=0))
            padded_pos_embs.append(torch.cat([zero_padding, pos_seq], dim=0))
            padded_ad_ids.append(torch.cat([id_padding, pos_ids], dim=0))
            
            mask = torch.ones([max_len])
            mask[:padding_len] = 0
            pad_mask.append(mask)

            neg_ids = self.generate_random_ids(unit_data_keys, set(ad_ids[idx]), current_seq_len)
            neg_emb_list = [self.unitid_data[nid]['embedding'] for nid in neg_ids]
            neg_seq = torch.tensor(neg_emb_list, dtype=torch.float32)
            
            # 负样本序列 Pre-padding
            padded_neg_embs.append(torch.cat([zero_padding, neg_seq], dim=0))

        return torch.stack(padded_embeddings), \
            torch.stack(padded_pos_embs), \
            torch.stack(padded_neg_embs), \
            torch.stack(pad_mask), \
            torch.stack(padded_ad_ids)

    def generate_random_ids(self, all_keys, exclude_id_set, count):
        res = []
        while len(res) < count:
            # 直接从原始键值（ID）列表中随机抽取
            random_id = random.choice(all_keys)
            # 确保抽取的原始 ID 不在当前用户的行为序列中
            if random_id not in exclude_id_set:
                res.append(random_id)
        return res


class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.gt_data = []
        self.test_data = []

        data_root = os.path.join(args.dataset_dir, "1w_tokenized_unitid")
        self.unitid_data,self.lenth_unit_data = self.load_unit(data_root)
        
        # # 读取test.txt文件
        # with open(f"{args.dataset_dir}/{args.test_file}", 'r') as f:
        #     for line in f:
        #         parts = line.strip().split('\t')
        #         user_id = int(parts[0].split("|")[0])
        #         ad_ids = list(map(int, parts[1].split()))
        #         self.test_data.append({'user_id': user_id, 'ad_ids': ad_ids})
        
        # # 读取test_gt.txt文件
        # with open(f"{args.dataset_dir}/{args.test_gt_file}", 'r') as f:
        #     for line in f:
        #         parts = line.strip().split('\t')
        #         user_id = int(parts[0].split("|")[0])
        #         ad_id = int(parts[1])
        #         self.gt_data[user_id] = ad_id

        # 暂时读取train.txt文件，取最后一位作为gt
        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split())) # 将训练数据最后一位暂时当测试数据
                ad_filter_ids = []
                for ad_id in ad_ids:
                    if ad_id in self.unitid_data:
                        ad_filter_ids.append(ad_id)
                self.test_data.append({'ad_ids': ad_filter_ids[:-1]})
                self.gt_data.append(ad_filter_ids[-1])
        print(f"test data loaded sucessfully ,{len(self.test_data)}")

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        sample = self.test_data[idx]
        ad_ids = sample["ad_ids"]
        ad_embeddings = []
        for ad_id in ad_ids:
            if ad_id in self.unitid_data:
                ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
            else:
                print(f"{ad_id} not in unit_map")

        # 截断    
        ad_embeddings = ad_embeddings[-self.args.maxlen:]

        return ad_embeddings,self.gt_data[idx],self.unitid_data[self.gt_data[idx]]["embedding"] # gt_embedding
    
    def load_unit(self, root_dir):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        
        file_list = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files]
        unitid_data = {}
        for f in file_list:
            safe_process_file(f,unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data,lenth_unit_data

    def collate_fn(self,batch):
        # 假设batch中的每个元素是一个元组 (user_id, ad_embeddings)
        ad_embeddings,gt,gt_embeddings = zip(*batch)
        gt_embeddings = torch.tensor(gt_embeddings,dtype=torch.float32)
        # 找到最长的ad_embeddings长度
        max_len = max(len(emb) for emb in ad_embeddings)
        
        # 初始化填充后的ad_embeddings和pad_mask
        padded_embeddings = []
        pad_mask = []
        
        for emb in ad_embeddings:
            emb_len = len(emb)
            padding_len = max_len - emb_len
            
            # 随机初始化填充向量，与 emb 具有相同的维度
            padding_vector = torch.randn([padding_len, self.args.emb_dim],dtype=torch.float32)
            
            # 拼接原始 embedding 和填充向量
            if padding_len:
                padded_emb = torch.cat([padding_vector,torch.tensor(emb)], dim=0)
            else:
                padded_emb = torch.tensor(emb,dtype=torch.float32)
            
            # 创建 pad_mask，1 表示原始数据，0 表示填充数据
            mask = torch.ones([max_len], dtype=torch.float32)
            mask[:padding_len] = 0
            
            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)
        
        padded_embeddings = torch.stack(padded_embeddings, dim=0)
        pad_mask = torch.stack(pad_mask, dim=0)
        
        # gt_data = []
        # for user_id in user_ids:
        #     gt_data.append(self.gt_data[user_id])
        gt_data = torch.tensor(gt)

        return padded_embeddings, pad_mask, gt_data,gt_embeddings
    