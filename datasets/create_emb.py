import numpy as np
import os
import pickle
from questions import dump_to_file


# Create emb matrix exactly for our vocab.
def create_embedding(emb_path, word2idx):

    with open(emb_path, 'r') as f:
        embs = f.readlines()
    vocab_size = len(word2idx)
    emb_dim = len(embs[0].split(' ')) - 1
    weights = np.zeros((vocab_size, emb_dim), dtype=np.float32)

    # Change the emb file to a dict.
    word2emb = {}
    for emb in embs:
        word = emb.split(' ')[0]
        feature = list(map(float, emb.split(' ')[1: ]))
        word2emb[word] = np.array(feature) # Change to numpy!

    # Get the emb matrix just for our vocab.
    for word, index in word2idx.items():
        if word in word2emb:
            weights[index] = word2emb[word]

    return weights, word2emb


# Execute only once to create emb matrix, after that you can get it from .pkl file directly.
if __name__ == '__main__':

    # Get and save the embedding matrix of our vocab.
    glove_file_path = '/data5/Kobeyond/data/glove/glove.6B.300d.txt'
    word2idx = pickle.load(open('/home/yangsj/GCN_VideoQA/data/pickle/word2idx3.pkl', 'rb')) ## MODIFY
    weights, word2emb = create_embedding(glove_file_path, word2idx)

    # Store the emb matrix in pkl file.
    dir_pkl_path = '/home/yangsj/GCN_VideoQA/data/pickle'
    dump_to_file(dir_pkl_path, 'emb_matrix3.pkl', weights)
