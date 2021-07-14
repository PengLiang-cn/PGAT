import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Answer module for different VideoQA tasks. The input of every module is the 
fusioned feature 'x' from: video, question, gcn graphs, memory.
"""

class CountClassifier(nn.Module):
    """
    The output of 'count' task is a number, which indicates the times a certain
    action occured in the video. So we regard 'count' as a regression problem.
    """
    def __init__(self, dim_h, dropout=None):
        super(CountClassifier, self).__init__()
        self.dim_h = dim_h
        self.dropout = dropout
        self.fc = nn.Linear(dim_h, 1)

    def forward(self, fusion):
        # fusion: [batch, dim_h]
        fusion = F.dropout(fusion, 0.5, training=self.training)
        logits = self.fc(fusion)
        # count = np.clip(np.round(logits), a_min=1, a_max=10)
        # count = logits.clamp(min=1, max=10)
        return logits.squeeze()
        # Never use 'ceil' or 'round', it will lead to bad result:
        # return torch.ceil(count)


class FrameqaClassifier(nn.Module):
    """
    The output of 'frameqa' task is an open-ended word, which is the answer of
    these four types of questions: Object/Number/Color/Location.
    We collect all the answers occured in the training set as the candidates.
    So we can regard 'frameqa' as a multi-label classification problem.
    """
    def __init__(self, dim_h, num_class, dropout=None):
        super(FrameqaClassifier, self).__init__()
        self.dim_h = dim_h
        self.num_class = num_class
        self.dropout = dropout
        self.fc = nn.Linear(dim_h, num_class) # [batch, num_class]

    def forward(self, fusion):
        # fusion: [batch, dim_h]
        fusion = F.dropout(fusion, 0.5, training=self.training)
        logits = self.fc(fusion)

        # No softmax for computational efficiency:
        # logits = F.softmax(self.fc(fusion), dim=1)
        return logits


class MultiChoiceClassifier(nn.Module):
    """
    Both in the tasks of 'action' and 'transition', the dataset provides 5 given
    choices and the correct choice number of every question.
    By combining fusion and Ans_mat, we get a confidece score for every answer.
    So we can regard it as a 5-label classification problem finally.
    """
    def __init__(self, dim_emb, dim_h, dropout=None):
        super(MultiChoiceClassifier, self).__init__()
        self.dim_emb = dim_emb
        self.dim_h = dim_h
        self.dropout = dropout

        self.fc1 = nn.Linear(dim_emb, dim_h)
        self.fc2 = nn.Linear(dim_h, 1)
        # self.fc3 = nn.Linear(dim_h, 1)


    def forward(self, fusion, ans_mat):
        # fusion: [batch, dim_h]
        # ans_mat: [batch, 5, ans_len(6), dim_emb]
        # Every is a phrase, attention or not???

        fusion = F.dropout(fusion, 0.5, training=self.training)
        ans = self.fc1(ans_mat).sum(2) # [batch, 5, dim_h]
        fusion = fusion.unsqueeze(1)
        logits = (self.fc2(fusion*ans)).squeeze() # [batch, 5]

        return logits

