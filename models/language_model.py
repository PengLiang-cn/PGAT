import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

DEVICE = 1

"""
Use Glove300d to embed the ques words into word-level emb-vectors.
Input:  Sequences of word indexs in the vocab.
Output: Sequences of emb vectors of the words in question.
"""
class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_dim, dropout=None):
        super(WordEmbedding, self).__init__()

        # padding_idx: when the word index equals 'vocab_size', we get an zero-emb for padding.
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=vocab_size)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dropout = dropout

    # Load the emb matrix exactly for our vocab from file.
    def init_embbeding(self):

        ### MODIFY ###
        emb_matrix = pickle.load(open('./data/pickle/emb_matrix3.pkl', 'rb'))
        ### MODIFY ###

        assert emb_matrix.shape == (self.vocab_size, self.emb_dim)
        # Do not init the emb at index[padding_idx]:
        self.emb.weight.data[: self.vocab_size] = torch.from_numpy(emb_matrix)

    def forward(self, x):
        # x: [batch, seq_len]
        # x_emb: [batch, seq_len, emb_dim]
        x_emb = self.emb(x)
        # x = self.dropout(x)
        return x_emb



"""
Use LSTM/GRU to encode the word-level emb features to a sentence-level feature.
Input:  Sequences of emb vectors of the words in question.
Output: A sentence-level emb feature of the total question.
"""
class QuestionEmbedding(nn.Module):

    def __init__(self, dim_in, dim_hid, num_layers, bi_direct, dropout=0, rnn_type='LSTM'):
        super(QuestionEmbedding, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.num_layers = num_layers
        self.bi_direct = bi_direct
        self.dropout = dropout
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn_type = rnn_type

        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(dim_in, dim_hid, num_layers=num_layers, bidirectional=bi_direct,
                           dropout=dropout, batch_first=True)

    def init_hidden(self, batch):
        bi = 2 if self.bi_direct else 1

        if self.rnn_type == 'LSTM':
            # Input: x, (h0, c0).   Output: x, (hN, cN).
            h0 = torch.zeros(self.num_layers * bi, batch, self.dim_hid).cuda()
            c0 = h0
            return (h0, c0)
        else: # GRU
            # Input: x, h0.   Output: x, hN.
            h0 = torch.zeros(self.num_layers * bi, batch, self.dim_hid).cuda()
            return h0

    def forward(self, x):

        # x: [batch, seq_len, dim_emb]
        batch = x.shape[0]
        h_state = self.init_hidden(batch)
        self.rnn.flatten_parameters()

        q_emb, state = self.rnn(x, h_state)  # [batch, seq_len, dim_hid * bi_direct]
        return q_emb, state



"""
Use word attention to highlight the important words in the sentence. And then we
can get a weight for every word emb, finally the weighted sum of all the word embs 
represents the sentence-level question embedding. 
"""
class Word_Attention(nn.Module):

    def __init__(self, dim_emb, dim_h):
        super(Word_Attention, self).__init__()
        self.dim_emb = dim_emb

        # self.fc1 = nn.Linear(dim_emb, 1)
        # self.fc2 = nn.Linear(dim_emb, dim_h)
        self.fc1 = nn.Linear(dim_h, 1)


    def forward(self, w_embs):
        # w_embs: [batch, seq_len, dim_emb]
        batch, seq_len, dim_emb = w_embs.shape

        w_embs = F.dropout(w_embs, 0.5, training=self.training)
        w = self.fc1(w_embs)
        w = torch.tanh(w.view(batch, seq_len))
        w = F.dropout(w, 0.5, training=self.training)

        w_att = F.softmax(w, 1) # [batch, seq_len]
        w_att = w_att.view(batch, seq_len, 1)

        q_emb = (w_att * w_embs).sum(1) # [batch, dim_emb]
        # q_emb = self.fc2(q_emb) # [batch, dim_h]
        return q_emb # now: [batch, dim_h]


