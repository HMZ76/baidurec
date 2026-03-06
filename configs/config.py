import torch
import os

class Args:
    def __init__(self):
        self.dataset_dir = "./data/w_data"
        self.unitid_file = "unitid.txt"
        self.train_file = "1w_train.txt"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.batch_size = 64
        self.lr = 0.001
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 10
        self.eval_stride = 5
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inference_only = False
        self.state_dict_path = None
        

args = Args()