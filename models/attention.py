import torch
import torch.nn as nn
import torch.nn.functional as F


"""Frame attention use q as the context vector, and finds the top-k relevent frames to 
    answer the question, and then we use the k video features to construct the Graph.
"""
class Attention(nn.Module):

    def __init__(self, q_dim, v_dim):
        super(Attention, self).__init__()
        self.q_dim = q_dim # [batch, dim_h]
        self.v_dim = v_dim # [batch, n_frames, dim_v]
        # self.k = k

        self.q_embedding = nn.Sequential(nn.Linear(q_dim, q_dim), nn.ReLU())
        self.v_embedding = nn.Sequential(nn.Linear(v_dim, q_dim), nn.ReLU())

        self.att = nn.Linear(q_dim, 1)

    def forward(self, v, q):

        if len(v.size()) == 3:
            q_encode = self.q_embedding(q).unsqueeze(1)
        elif len(v.size()) == 4:
            q_encode = self.q_embedding(q).unsqueeze(1).unsqueeze(2)
        else:
            q_encode = self.q_embedding(q).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        v_encode = self.v_embedding(v)
        fusion = v_encode * q_encode

        map = self.att(fusion)
        map = F.softmax(map, -2)

        att_v = (map * v).sum(-2)

        return att_v
