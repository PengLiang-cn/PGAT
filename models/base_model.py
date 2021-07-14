import torch
import torch.nn as nn
import torch.nn.functional as F
from models.language_model import WordEmbedding, QuestionEmbedding, Word_Attention
from models.answer_model import *
from models.gcn_model import GCN
from models.attention import Attention


def position_encoding_init(v_len, v_emb_dim):

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / v_emb_dim) for j in range(v_emb_dim)]
        if pos != 0 else np.zeros(v_emb_dim) for pos in range(v_len)
    ])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def select_trajectory_region(region):
    ### input:  (batch_size, N, K, dim)
    ### output: (batch_size, N, K, dim)

    assert len(region.size()) == 4
    batch_size, N, K, dim = region.size()

    #########   forward
    anchor = region[:, 0, :, :]
    anchor = anchor.unsqueeze(1)
    anchor = anchor.repeat(1, N-1, 1, 1)      #### (batch_size, N-1, K, dim)


    other = region[:, 1:, :, :]               #### (batch_size, N-1, K, dim)
    # other = other.unsqueeze(2)
    # other = other.repeat(1, 1, K, 1, 1)

    # dist = torch.sqrt(torch.sum((anchor-other)**2, -1)) ##### (batch_size, N-1, K, K)

    dist = torch.matmul(anchor, other.transpose(2,3))     ##### (batch_size, N-1, K, K)

    index = torch.argmax(dist, -1)                      ##### (batch_size, N-1, K)
    index = index.unsqueeze(3).repeat(1, 1, 1,dim)

    orign_other = region[:, 1:, :, :]
    forward_other = torch.gather(orign_other, 2, index)
    forward = torch.cat([region[:, 0, :, :].unsqueeze(1), forward_other], 1)

    #########   backward

    anchor = region[:, -1, :, :]
    anchor = anchor.unsqueeze(1)
    anchor = anchor.repeat(1, N-1, 1, 1)      #### (batch_size, N-1, K, dim)

    other = region[:, :-1, :, :]  #### (batch_size, N-1, K, dim)
    # other = other.unsqueeze(2)
    # other = other.repeat(1, 1, K, 1, 1)

    dist = torch.matmul(anchor, other.transpose(2, 3))  ##### (batch_size, N-1, K, K)

    index = torch.argmax(dist, -1)  ##### (batch_size, N-1, K)
    index = index.unsqueeze(3).repeat(1, 1, 1, dim)

    orign_other = region[:, :-1, :, :]
    backward_other = torch.gather(orign_other, 2, index)
    backward = torch.cat([backward_other, region[:, -1, :, :].unsqueeze(1)], 1)

    return forward, backward


""" A simple base model combined all parts of the VideoQA structure, from which 
we can easily get the results by: 'output = model(input)' """
class BaseModel(nn.Module):

    def __init__(self, task, w_emb, q_emb, w_att, b_att, f_att, b_gcn, b_gcn_2, b_gcn_3, f_gcn, \
                            classifier, f_net, v_net, q_net, m_net, m_gcn, m_att):
        super(BaseModel, self).__init__()
        self.task = task # ques type

        # Simple demo to test:
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.w_att = w_att
        self.f_gcn = f_gcn
        self.b_gcn = b_gcn
        self.b_gcn_2 = b_gcn_2
        self.b_gcn_3 = b_gcn_3
        self.f_att = f_att
        self.b_att = b_att
        self.f_net = f_net
        self.v_net = v_net
        self.q_net = q_net
        self.m_net = m_net
        self.m_gcn = m_gcn
        self.m_att = m_att
        self.clasifier = classifier # different from task to task


    def forward(self, regional_feats, boxes, frame_feats, motion_feats, ques_idx, label, choices=None):


        # ---------------- Ques Attention ----------------
        words_emb = self.w_emb(ques_idx) # batch × seq_len × dim_emb
        words_emb_size = words_emb.size(-1)
        batch_size = words_emb.size(0)
        words_emb = self.q_emb(words_emb)[0] # LSTM -> batch × seq_len × dim_h
        q = self.w_att(words_emb) # batch × dim_h

        # ---------------- Regional Video Feats ----------------
        new_region_feat = torch.cat([regional_feats, boxes], dim=3)                                 # batch × 16 × 8 × (2048+4)
        spa_region_feat = new_region_feat.view(batch_size, 8, 4, 20, -1)
        spa_region_gcn = self.b_gcn(spa_region_feat, q) + spa_region_feat     #### spatial only

        tra_region_feat = new_region_feat.view(batch_size*8, 4, 20, -1)
        tra_forward, tra_backward = select_trajectory_region(tra_region_feat)
        # tra_forward = tra_forward.view(batch_size, 20, 4*8, -1)
        # tra_backward = tra_backward.view(batch_size, 20, 4*8, -1)
        tra_forward = tra_forward.view(batch_size, 8, 20, 4, -1)
        tra_backward = tra_backward.view(batch_size, 8, 20, 4, -1)

        tra_forward = self.b_gcn_2(tra_forward, q) + tra_forward  #### spatial only
        tra_forward = tra_forward.view(batch_size, 8, 4, 20, -1)

        tra_backward = self.b_gcn_3(tra_backward, q) + tra_backward  #### spatial only
        tra_backward = tra_backward.view(batch_size, 8, 4, 20, -1)

        # region_gcn = spa_region_gcn + tra_backward + tra_forward
        # region_gcn = new_region_feat
        spa_region = self.b_att(spa_region_gcn, q) 			                                            # batch × 8 x 2 × (2048+4)
        tra_backward = self.b_att(tra_backward, q)
        tra_forward = self.b_att(tra_forward, q)

        v_region = spa_region + tra_backward + tra_forward



        # ---------------- Frame Video Feats ----------------
        new_frame_feats = self.f_net(v_region) + frame_feats.view(batch_size, 8, 4, -1)          # batch × 16 × (2048)
        frame_gcn = self.f_gcn(new_frame_feats, q) + new_frame_feats
        v_frame = self.f_att(frame_gcn, q)                                                          # batch × 8 x (2048)

        # ---------------- motion Video Feats ----------------
        new_motion_feats = self.m_net(v_frame) + motion_feats
        motion_gcn = self.m_gcn(new_motion_feats, q) + new_motion_feats
        v_motion = self.m_att(motion_gcn, q)

        v = v_motion
        joint = self.v_net(v) * self.q_net(q)

        # ---------------- Answer Module ----------------
        # Combines the 5 choices to infer:
        if self.task in ['action', 'transition']:
            if not torch.is_tensor(choices):
                print('The 5 choices are required!')
                return
            else:
                choices_emb = self.w_emb(choices).view(-1, 6, words_emb_size)          # [batch, 5, ans_len(6), dim_emb]
                choices_emb = self.q_emb(choices_emb)[0].view(batch_size, 5, 6, -1)
                # joint = choices_emb.sum(2).sum(1)
                logits = self.clasifier(joint, choices_emb)
        else:
            logits = self.clasifier(joint) # differs from task to task
        return logits




def build_baseline(task, vocab_size, num_f, dim_vf, num_obj, dim_vr, dim_emb, dim_h, num_class=None):

    dim_vr = dim_vr + 4
    # dim_vf = dim_vr

    word_emb = WordEmbedding(vocab_size, dim_emb)
    word_emb.init_embbeding()

    ques_emb = QuestionEmbedding(dim_emb, dim_h, num_layers=1, bi_direct=False, rnn_type='LSTM')
    w_att = Word_Attention(dim_emb, dim_h)

    box_att = Attention(dim_h, dim_vr)
    box_gcn = GCN(dim_h, dim_vr)
    box_gcn_2 = GCN(dim_h, dim_vr)
    box_gcn_3 = GCN(dim_h, dim_vr)
    # box_mlp = nn.Sequential(nn.Linear(dim_vr*2, dim_vr), nn.ReLU())

    frame_mlp = nn.Sequential(nn.Linear(dim_vr, dim_vf), nn.ReLU())
    frame_att = Attention(dim_h, dim_vf)
    frame_gcn = GCN(dim_h, dim_vf)

    motion_mlp = nn.Sequential(nn.Linear(dim_vf, dim_vf), nn.ReLU())
    motion_att = Attention(dim_h, dim_vf)
    motion_gcn = GCN(dim_h, dim_vf)

    v_fusion = nn.Sequential(nn.Linear(dim_vf, dim_h), nn.ReLU())
    q_fusion = nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())

    if task in ['action', 'transition']:
        classifier = MultiChoiceClassifier(dim_h, dim_h, dropout=None)
    elif task == 'frameqa':
        classifier = FrameqaClassifier(dim_h, num_class)
    else: # count
        classifier = CountClassifier(dim_h)

    # Use every components to form the base model.
    base_model = BaseModel(task, word_emb, ques_emb, w_att, box_att, frame_att, box_gcn, box_gcn_2, box_gcn_3,\
                                                    frame_gcn, classifier, frame_mlp, v_fusion, q_fusion, motion_mlp,\
                                                    motion_gcn,  motion_att)
    return base_model

