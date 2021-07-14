import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, dim_h, dim_v, dropout=0.5):
        super(GCN, self).__init__()

        self.dim_v = dim_v
        self.dim_h = dim_h

        self.mlp_1 = nn.Sequential(nn.Linear(dim_v, dim_h), nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Linear(dim_v, dim_h), nn.ReLU())
        self.q_mlp = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())

    def forward(self, x, q, need_q=True):

        if len(x.size()) == 3:
            batch, num_clip, dim_v = x.size()
            expand_size = 1
        elif len(x.size()) == 4:
            batch, num_clip, num_frame, dim_v = x.size()
            expand_size = num_clip
        else:
            batch, num_clip, num_frame, num_obj, dim_v = x.size()
            expand_size = num_clip * num_frame

        x_c = x.view(-1, x.size(-2), dim_v)

        if need_q:
            q_emb = self.q_mlp(q).repeat(expand_size, 1).unsqueeze(1)
            x_emb_1 = self.mlp_1(x_c) * q_emb
            x_emb_2 = self.mlp_2(x_c) * q_emb
        else:
            x_emb_1 = self.mlp_1(x_c)
            x_emb_2 = self.mlp_2(x_c)

        adj = torch.matmul(x_emb_1, x_emb_2.transpose(1, 2))
        adj = F.softmax(adj, 2)

        gcn_x = torch.matmul(adj, x_c)

        if len(x.size()) == 3:
            return gcn_x.view(batch, num_clip, dim_v)
        elif len(x.size()) == 4:
            return gcn_x.view(batch, num_clip, num_frame, dim_v)
        else:
            return gcn_x.view(batch, num_clip, num_frame, num_obj, dim_v)