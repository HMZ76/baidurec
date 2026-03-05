import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from datetime import date
from tqdm import tqdm
import random
import torch.nn as nn
import torch.nn.functional as F


# demo数据集的目录包含1w_train.txt文件和1w_unitid_title_emb目录
# 取1w条数据训练,实际训练和测试的数据集要远远大于demo使用的数据集
# 对于长度为n的广告序列，只取前n-1个序列作为训练集，最后一位作为测试集。前向pad构建batch

def read_data(f,unitid_data):
    with open(f, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            ad_id = int(parts[0])
            embedding = list(map(np.float32, parts[2].split(',')))
            unitid_data[ad_id] = {'embedding': embedding}

def safe_process_file(f,unitid_data):
    try:
        read_data(f,unitid_data)
    except Exception as e:
        print(e)

class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.unitid_data,self.lenth_unit_data = self.load_unit()
        self.train_data = []
        
        # 读取train.txt文件
        cnt_null_unit = 0
        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split()))[:-1] # 将训练数据最后一位暂时当测试数据
                ad_filter_ids = []
                for ad_id in ad_ids:
                    if ad_id in self.unitid_data:
                        ad_filter_ids.append(ad_id)
                    else:
                        cnt_null_unit += 1
                self.train_data.append({'ad_ids': ad_filter_ids})
                
        print(f"train.txt loaded sucessfully ,{len(self.train_data)},{cnt_null_unit} units which is not in unit_map")
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx]
        ad_ids = sample['ad_ids']
        
        ad_embeddings = []
        for idx,ad_id in enumerate(ad_ids):
            if ad_id in self.unitid_data:
                ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
            else:
                print(f"{ad_id} not in unit_map")
        lenth = len(ad_embeddings)
        return ad_embeddings[lenth-self.args.maxlen:],ad_ids[lenth-self.args.maxlen:]
    
    def load_unit(self):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        root_dir = "./data/data322235/w_data/1w_tokenized_unitid"
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
        # print(batch)
        # 假设batch中的每个元素是一个元组 (user_id, ad_embeddings)
        ad_embeddings,ad_ids = zip(*batch)
        # 找到最长的ad_embeddings长度
        max_len = max(len(emb[:-1]) for emb in ad_embeddings)
        
        # 初始化填充后的ad_embeddings和pad_mask
        padded_embeddings = []
        pad_mask = []
        padded_pos_embs = []
        padded_neg_embs = []
        padded_ad_ids = []
        
        for idx,emb in enumerate(ad_embeddings):
            emb_len = len(emb[:-1])
            ad_ids_vector = torch.tensor(ad_ids[idx][1:],dtype=torch.float32)
            padding_len = max_len - emb_len
            if padding_len:
                # 随机初始化填充向量，与 emb 具有相同的维度
                padding_vector = torch.randn([padding_len, self.args.emb_dim],dtype=torch.float32)
                padding_ad_vector = torch.full([padding_len],0,dtype=torch.float32) # pad id 0
                # 拼接原始 embedding 和填充向量
                padded_emb = torch.cat([padding_vector,torch.tensor(emb[:-1],dtype=torch.float32)], dim=0)
                padded_ad_ids_vector = torch.cat([padding_ad_vector,ad_ids_vector],dim=0)
            else:
                padded_ad_ids_vector = ad_ids_vector
                padded_emb = torch.tensor(emb[:-1],dtype=torch.float32)
            padded_ad_ids.append(padded_ad_ids_vector)
            # 创建 pad_mask，1 表示原始数据，0 表示填充数据
            mask = torch.ones([max_len], dtype=torch.float32)
            mask[:padding_len] = 0
            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)
            if padding_len:
                padded_pos_emb = torch.cat([padding_vector,torch.tensor(emb[1:],dtype=torch.float32)], dim=0)
            else:
                padded_pos_emb = torch.tensor(emb[1:],dtype=torch.float32)
            unit_data_map_ids = list(self.unitid_data.keys())
            random_neg_ids = self.generate_random_numbers(0, self.lenth_unit_data-1, [ad_ids[idx][1:]], emb_len)
            # 随机负例
            random_neg_emb = torch.tensor([self.unitid_data[unit_data_map_ids[i]]['embedding'] for i in random_neg_ids],dtype=torch.float32)
            if padding_len:
                padded_neg_emb = torch.cat([padding_vector,random_neg_emb], dim=0)
            else:
                padded_neg_emb = random_neg_emb

            padded_pos_embs.append(padded_pos_emb)
            padded_neg_embs.append(padded_neg_emb)
        
        padded_embeddings = torch.stack(padded_embeddings, dim=0)
        pad_mask = torch.stack(pad_mask, dim=0)
        padded_pos_embeddings = torch.stack(padded_pos_embs,dim=0)
        padded_neg_embeddings = torch.stack(padded_neg_embs,dim=0)
        padded_ad_ids = torch.stack(padded_ad_ids,dim=0)

        return padded_embeddings, padded_pos_embeddings, padded_neg_embeddings, pad_mask, padded_ad_ids # ad_ids

    def generate_random_numbers(self,start, end, exceptions, count):
        """
        生成count个在[start, end]范围内的随机数，但不能是exceptions列表中的值。
        
        :param start: 随机数的起始范围
        :param end: 随机数的结束范围
        :param exceptions: 不能生成的数值列表
        :param count: 需要生成的随机数数量
        :return: 生成的随机数列表
        """
        random_numbers = []
        while len(random_numbers) < count:
            num = random.randint(start, end)
            if num not in exceptions:
                random_numbers.append(num)
                # exceptions.append(num)  # 将已生成的随机数添加到exceptions列表中，以避免重复
        return random_numbers

# 测试集的构建

class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.gt_data = []
        self.test_data = []
        self.unitid_data,self.lenth_unit_data = self.load_unit()
        
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
        
        return ad_embeddings,self.gt_data[idx],self.unitid_data[self.gt_data[idx]]["embedding"] # gt_embedding
    
    def load_unit(self):
        import time
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        root_dir = "data/data322235/w_data/1w_unitid_title_emb"
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
    

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.permute(0, 2, 1))))))
        outputs = outputs.permute(0, 2, 1)  # as Conv1d requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.dev = args.device
        # self.item_embedding = torch.nn.Embedding(1 + self.lenth_unit_id,embedding_dim=self.args.hidden_units,padding_idx=self.lenth_unit_id)
        self.pos_emb = torch.nn.Embedding(num_embeddings=args.maxlen+1, embedding_dim=args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(normalized_shape=args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(normalized_shape=args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(normalized_shape=args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units=args.hidden_units, dropout_rate=args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, seqs , mask): 
        # print(f"seqs.shape:{seqs.shape},mask.shape:{mask.shape}")
        pos = torch.tensor(np.tile(np.arange(1, seqs.shape[1] + 1), [seqs.shape[0], 1]),dtype=torch.float32).to(self.dev)
        # print(f"pos.shape:{pos.shape}")
        mask = mask.to(self.dev)
        pos *= mask
        seqs = seqs.to(self.dev)
        seqs += self.pos_emb(torch.tensor(pos, dtype=torch.int64).to(self.dev))
        seqs = self.emb_dropout(seqs)
        # print(f"pos emb seqs.shape:{seqs.shape}")

        tl = seqs.shape[1]  # time dim len for enforce causality
        # 修复：attention_mask 需要移动到设备上
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool), diagonal=0).to(self.dev)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            
            Q = Q.to(self.dev)
            
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            # print(f"attention layer {i}:seqs.shape{seqs.shape}")

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            # print(f"forward layer {i}:seqs.shape{seqs.shape}")

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self,seqs, mask, pos_seqs, neg_seqs):  # for training        
        logits = self.log2feats(seqs,mask)  # user_ids hasn't been used yet
        # print(f"log_feats:{log_feats.shape},pos_seqs.shape:{pos_seqs.shape},neg_seqs.shape:{neg_seqs.shape}")
        # pos_logits = torch.sum(log_feats * pos_seqs, axis=-1)
        # neg_logits = torch.sum(log_feats * neg_seqs, axis=-1)
        # print(f"pos_logits.shape:{pos_logits.shape},neg_logits.shape:{neg_logits.shape}")

        # return pos_logits, neg_logits  # pos_pred, neg_pred
        return logits  # B * S * D

    def predict(self, seqs, mask, item_embs):  # for inference
        log_feats = self.log2feats(seqs,mask)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        logits = torch.matmul(final_feat, item_embs.transpose(0, 1))
        return logits  # preds  # (U, I)

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
    
# 配置文件
class Args():
    def __init__(self):
        self.dataset_dir = "./data/data322235/w_data" # root directory containing the datasets
        # self.train_dir = "train_data" # subdirectory containing the training data
        self.unitid_file = "unitid.txt"
        self.train_file = "1w_train.txt"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.batch_size = 32
        self.lr = 0.0001
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 3
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "cuda"
        self.inference_only = False
        # self.state_dict_path = "2025_03_27/SASRec.epoch=3.lr=0.0001.layer=2.head=1.hidden=1024.maxlen=200.pth"
        self.state_dict_path = None

args = Args()
with open(os.path.join(args.dataset_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


def evaluate(model):
    print("开始加载测试数据...")
    dataset = TestDataset(args)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    print("数据加载完成")
    # 全库 embedding
    # item_embs = torch.to_tensor([v["embedding"] for k,v in unitid_data.items()]) 
    # sigmoid = torch.nn.Sigmoid()
    sf = torch.nn.Softmax(dim=-1)
    test_result = {"recall@5":0,"recall@10":0,"ndcg@5":0,"ndcg@10":0}
    with torch.no_grad():
        cnt_batch = 0
        for padded_embeddings, pad_mask, gt,gt_embeddings in tqdm(dataloader):
            # 修复：将所有输入张量移动到模型所在设备
            padded_embeddings = padded_embeddings.to(model.dev if hasattr(model, 'dev') else args.device)
            pad_mask = pad_mask.to(model.dev if hasattr(model, 'dev') else args.device)
            gt_embeddings = gt_embeddings.to(model.dev if hasattr(model, 'dev') else args.device)
            gt = gt.to(model.dev if hasattr(model, 'dev') else args.device)
            
            # logits = model.predict(padded_embeddings,pad_mask,item_embs) # 全库检索
            logits = model.predict(padded_embeddings,pad_mask,gt_embeddings) # gt 向量里检索
            probs = sf(logits)
            topk_values, topk_indices = torch.topk(probs, k=10, dim=-1)
            # 修复：gt 需要与 topk_indices 在同一设备上
            gt = torch.arange(0, gt.shape[0], dtype=torch.int64, device=topk_indices.device)
            def ndcg(top_k,topk_indices,gt):
                # 计算 DCG@k
                ranks = torch.arange(1, top_k + 1, dtype=torch.float32, device=topk_indices.device)  # 排名：[1, 2, ..., k]
                dcg = torch.sum(torch.where(topk_indices == gt.unsqueeze(-1), 1.0 / torch.log2(ranks + 1), torch.zeros_like(topk_indices, dtype=torch.float32, device=topk_indices.device)), dim=-1)
                # IDCG 为 0
                # # 计算 IDCG@k（假设正确标签在第一位）
                # idcg = 1.0 / torch.log2(2)
                # # 计算 NDCG@k
                # ndcg = dcg / idcg
                test_result[f"ndcg@{top_k}"] += torch.mean(dcg).item()
            def recall(top_k,topk_indices,gt):
                # 计算 recall@k
                score = torch.sum(torch.where(topk_indices == gt.unsqueeze(-1), 1, 0), dim=-1)
                test_result[f"recall@{top_k}"] += torch.mean(score.to(torch.float32)).item()
            ndcg(10,topk_indices,gt)
            ndcg(5,topk_indices[:,:5],gt)
            recall(10,topk_indices,gt)
            recall(5,topk_indices[:,:5],gt)
            cnt_batch += 1
    test_result = {k:v/cnt_batch for k,v in test_result.items()}
    return test_result

model = SASRec(args).to('cuda') # no ReLU activation in original SASRec implementation?
model.to('cuda')
# 定义一个 XavierNormal 初始化器
from torch.nn.init import xavier_normal_

for name, param in model.named_parameters():
    try:
        if param.dim() >= 2:
            xavier_normal_(param)
    except:
        print(f"{name} xaiver 初始化失败")
        pass  # 忽略初始化失败的层

# model.pos_emb.weight.data[0, :] = 0
# model.item_emb.weight.data[0, :] = 0

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location='cpu'))
    except:  
        print('failed loading state_dicts, pls check file path: ')
        

if args.inference_only:
    model.eval()
    t_test = evaluate(model)
    print(t_test)
    print("Done")
else:
    print("开始加载训练数据...")
    dataset = TrainDataset(args)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    print("数据加载完成")

    criterion = CustomContrastiveLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = lr_scheduler(adam_optimizer, gamma=0.96)
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    step = 0
    accumulated_step = 0
    print("开始训练")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids in tqdm(dataloader): 
            padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids = \
                  padded_embeddings.to('cuda'), \
                    padded_pos_emb.to('cuda'), \
                        padded_neg_emb.to('cuda'), pad_mask.to('cuda'), ad_ids.to('cuda')
            logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
            loss = criterion(logits,padded_pos_emb,pad_mask,ad_ids)
            loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
            step += 1
        scheduler.step()
        # 打印当前学习率
        print(f'Epoch {epoch}, Current learning rate: {adam_optimizer.param_groups[0]["lr"]}')
        today = date.today()
        day = today.strftime("%Y_%m_%d") # 2023_10_05
        folder = "/home/aistudio" + "/" + day
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(folder, fname))
        t0 = time.time()
        model.train()
    print("Done") 