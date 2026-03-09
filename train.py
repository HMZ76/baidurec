import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
from datetime import datetime

from configs.config import args
from training.dataset import TrainDataset, TestDataset
from models.SASRec import SASRec
from training.loss import CustomContrastiveLoss
from utils.evaluate import evaluate 

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # 1. 基础配置与模型初始化
    device = torch.device(args.device)
    model = SASRec(args).to(device)

    current_time = datetime.now().strftime('%Y-%m%d-%H%M')
    log_dir = os.path.join('runs', f'SASRec_{args.num_blocks}L_{args.num_heads}H_{current_time}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    if args.state_dict_path and os.path.exists(args.state_dict_path):
        model.load_state_dict(torch.load(args.state_dict_path, map_location=device))
        print(f"Loaded checkpoint from {args.state_dict_path}")

    # 2. 数据准备
    print("Loading datasets...")
    test_dataset = TestDataset(args)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=test_dataset.collate_fn
    )

    if args.inference_only:
        results = evaluate(model, test_loader, device)
        print(f"Test Results: {results}")
        return

    train_dataset = TrainDataset(args)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=train_dataset.collate_fn, 
        shuffle=True
    )

    # 3. 优化器与损失函数
    criterion = CustomContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # 4. 训练与验证循环
    best_ndcg = 0.0
    global_step = 0 # 用于记录总步数
    print("Starting training...")
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        
        for batch in pbar:
            padded_embeddings, pos_emb, neg_emb, mask, ad_ids = [x.to(device) for x in batch]
            
            logits = model(padded_embeddings, mask, pos_emb, neg_emb)
            loss = criterion(logits, pos_emb, mask, ad_ids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- 记录 Batch 级别的 Loss ---
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # --- 记录 Epoch 级别的指标 ---
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch % args.eval_stride == 0:
            metrics = evaluate(model, test_loader, device)
            print(f"\n[Epoch {epoch}] Avg Loss: {avg_loss:.4f} | Metrics: {metrics}")

            for k, v in metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, epoch)

        save_dir = f"./checkpoints/{args.num_blocks}layer_{args.num_heads}head"
        os.makedirs(save_dir, exist_ok=True)
        
        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            torch.save(model.state_dict(), os.path.join(save_dir, "sasrec_best.pth"))
            print(f"New best NDCG@10: {best_ndcg:.4f}, model saved.")

        torch.save(model.state_dict(), os.path.join(save_dir, f"sasrec_epoch_{epoch}.pth"))
    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    main()