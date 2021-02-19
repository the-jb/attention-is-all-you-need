import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, n=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1, e_ls=0.1):
        super(TransformerModel, self).__init__()
        print(f"Init TransformerModel : {n=}, {d_model=}, {d_ff=}, {h=}, {d_k=}, {d_v=}, {p_drop=}, {e_ls=}")
