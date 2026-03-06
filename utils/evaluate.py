import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval() # 确保开启 Eval 模式（关闭 Dropout 等）
    test_result = {"recall@5": 0, "recall@10": 0, "ndcg@5": 0, "ndcg@10": 0}
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # 这里的解包需要对应 TestDataset.collate_fn 的输出
        padded_embeddings, pad_mask, gt_ids, gt_embeddings = [x.to(device) for x in batch]
        
        # 1. 获取模型预测结果 (Batch_Size, Num_Candidates)
        # 这里的 gt_embeddings 作为候选集进行检索
        logits = model.predict(padded_embeddings, pad_mask, gt_embeddings) 
        
        # 2. 计算 Top-K
        batch_size = logits.size(0)
        total_samples += batch_size
        _, topk_indices = torch.topk(logits, k=10, dim=-1)
        
        # 在此检索场景下，正确答案在 gt_embeddings 中的索引就是 torch.arange(batch_size)
        targets = torch.arange(batch_size, device=device).unsqueeze(-1)
        
        # 3. 计算 Recall
        hits = (topk_indices == targets) # (Batch_Size, 10)
        test_result["recall@10"] += hits.any(dim=-1).float().sum().item()
        test_result["recall@5"] += hits[:, :5].any(dim=-1).float().sum().item()
        
        # 4. 计算 NDCG
        # 找到目标在 topk 中的排名 (1-based)，若不在 topk 中则为 0
        hit_ranks = (hits == True).nonzero()
        if hit_ranks.size(0) > 0:
            # hit_ranks 格式: [样本索引, 排名索引]
            ranks = hit_ranks[:, 1] + 1 
            all_ndcg = 1.0 / torch.log2(ranks + 1)
            
            # 分别统计 NDCG@10 和 NDCG@5
            test_result["ndcg@10"] += all_ndcg.sum().item()
            test_result["ndcg@5"] += all_ndcg[ranks <= 5].sum().item()

    # 计算全局平均值
    final_metrics = {k: v / total_samples for k, v in test_result.items()}
    return final_metrics