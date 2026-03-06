import torch
import torch.nn as nn
import numpy as np

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout(self.conv2(self.relu(self.dropout(self.conv1(inputs.permute(0, 2, 1))))))
        return outputs.permute(0, 2, 1) + inputs

class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dev = args.device
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(args.hidden_units, args.num_heads, dropout=args.dropout_rate, batch_first=True)
            for _ in range(args.num_blocks)
        ])
        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(args.hidden_units) for _ in range(args.num_blocks)])
        self.forward_layers = nn.ModuleList([PointWiseFeedForward(args.hidden_units, args.dropout_rate) for _ in range(args.num_blocks)])
        self.forward_layernorms = nn.ModuleList([nn.LayerNorm(args.hidden_units) for _ in range(args.num_blocks)])
        self.last_layernorm = nn.LayerNorm(args.hidden_units)

        self._init_weights()

    def _init_weights(self):
        """内部权重初始化逻辑"""
        for name, param in self.named_parameters():
            try:
                # 仅对维度大于等于 2 的权重使用 Xavier Normal 初始化
                if param.dim() >= 2:
                    torch.nn.init.xavier_normal_(param)
            except Exception:
                pass

    def log2feats(self, seqs, mask):
        seq_len = seqs.shape[1]
        pos = torch.arange(1, seq_len + 1).unsqueeze(0).repeat(seqs.shape[0], 1).to(self.dev)
        pos = (pos * mask).long()
        

        seqs = seqs.to(self.dev) + self.pos_emb(pos)
        seqs = self.emb_dropout(seqs)

        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_out, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attn_mask)
            seqs = Q + mha_out
            seqs = self.forward_layers[i](self.forward_layernorms[i](seqs))

        return self.last_layernorm(seqs)


    def forward(self, seqs, mask, *args):
        return self.log2feats(seqs, mask)

    def predict(self, seqs, mask, item_embs):
        feats = self.log2feats(seqs, mask)[:, -1, :]
        return torch.matmul(feats, item_embs.transpose(0, 1))
    