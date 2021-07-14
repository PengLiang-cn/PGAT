# from questions import QuestionDataset  # only for unit test
import numpy as np
import os
import h5py
import csv
import base64
from torch.utils.data import Dataset
import sys
import torch
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

csv.field_size_limit(sys.maxsize)

def get_video_feats(hdf5_path):
    if not os.path.exists(hdf5_path):
        print('Subsampled feature file does not exist!')
    else:
        return h5py.File(hdf5_path, 'r')


def subsample_frames(save_path, video_feats, max_frames=16):
    if os.path.exists(save_path):
        print('Subsample file alreadly exists!')
        return

    video_feats_new = h5py.File(save_path, 'w')
    print('Subsample start...')
    for idx in video_feats:
        # video feat: [n_frame, 4096]
        n_frames = video_feats[idx].shape[0]

        print('feature%s start!' % idx)
        if n_frames >= max_frames:
            # Subsample frames centered at each time step:
            step = n_frames // max_frames
            remain = n_frames % max_frames
            feat_new = video_feats[idx][step - 1: n_frames - remain: step] # Crucial, important and critical !!!!
        else:
            # Pad the last frame until the length is 'n_frames':
            feat_new = np.zeros((max_frames, 2048), dtype=np.float32)
            feat_new[: n_frames] = video_feats[idx]
            last_frame = video_feats[idx][n_frames - 1]
            feat_new[n_frames: ] = np.tile(last_frame, (max_frames - n_frames, 1))  # right padding

        video_feats_new.create_dataset(idx, data=feat_new)
    video_feats_new.close()
    print('Subsample finished...')


def decode_base64(data):
    """Decode base64, padding being optional.
    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    data = bytes(data, encoding='utf8')
    # print(type(data))
    # print('len', len(data))
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += b'='* missing_padding
    data = base64.decodestring(data)
    # print(type(data))
    return data

""" A costum dataset class for VideoQA, which will be directly sent to dataloader.
video_feats: a h5py.File object, we can fetch the features of the question by video_id, 
             simply using 'video_feats[vid_id]'.
questions: a specific type of questions, we will get different returned values
           from 'dataset[i]', depending on ques type. """
class VideoQANew(Dataset):

    def __init__(self, ques_type, app_video_feats, mot_id_to_index, tsv_dir, questions, vocab_size):
        self.ques_type = ques_type
        self.app_video_feats = app_video_feats # Subsampled already
        self.mot_id_to_index = mot_id_to_index

        self.tsv_dir = tsv_dir # Root directory for faster R-CNN features.
        self.questions = questions # A specific type of question set.

        self.vocab_size = vocab_size
        self.dim_v = app_video_feats['99998'].shape[1]

    def __len__(self):
        return len(self.questions)

    def get_regional_features(self, video_name):
        region_feats = np.zeros((16, 8, 2048), dtype=np.float32)
        region_boxes = np.zeros((16, 8, 4), dtype=np.float32)

        tsv_path = os.path.join(self.tsv_dir, video_name + '.tsv')
        with open(tsv_path, "r+") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t', fieldnames=FIELDNAMES)
            # sortedlist = sorted(reader, key=lambda x: x[5], reverse=True)
            for i, item in enumerate(reader):
                feats = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape((8, -1)) # 8 × 2048
                boxes = np.frombuffer(decode_base64(item['boxes']), dtype=np.float32).reshape((8, -1)) # 8 × 4

                # frame_id = int(item['image_id'].split('__')[1])  # id: 0 ~ 15
                # print(feats.shape)
                # print(boxes.shape)

                region_feats[i] = feats
                region_boxes[i] = boxes

        return region_feats, region_boxes


    def __getitem__(self, i):
        ques = self.questions[i]

        video_name = ques['gif_name']
        region_feats, region_boxes = self.get_regional_features(video_name)

        video_id = ques['vid_id']
        app_video_feats = np.array(self.app_video_feats[video_id]) # 16 × 2048
        motion_index = self.mot_id_to_index[video_id]
        with h5py.File(os.path.join('./data/' , self.ques_type + '_motion_feat.h5'), 'r') as f_motion:
            mot_video_feats = f_motion['resnext_features'][motion_index]  # (8, 2048)

        # mot_video_feats = np.array(self.mot_video_feats[video_id])  # 16 × 4096
        ques_words_idx = np.array(ques['words_idx_UNK']) # Not emb features, but index seq.

        if self.ques_type in ['action', 'transition']:
            ans_len = len(ques['a1_UNK_idx'])
            ans_idx = np.zeros((5, ans_len), dtype=np.int)
            for i in range(1, 6):
                ans_name = 'a' + str(i) + '_UNK_idx'
                ans_idx[i - 1] = np.array(ques[ans_name])
            label = np.array(int(ques['answer'])) # choice number
            # v, q, ans, label:
            return torch.from_numpy(region_feats), torch.from_numpy(region_boxes),\
                   torch.from_numpy(app_video_feats), torch.from_numpy(mot_video_feats), \
                   torch.from_numpy(ques_words_idx), torch.from_numpy(ans_idx), torch.from_numpy(label)

        elif self.ques_type == 'count':
            # No choices, only a count number as label:
            label = np.array(int(ques['answer']))
            return torch.from_numpy(region_feats), torch.from_numpy(region_boxes), \
                   torch.from_numpy(app_video_feats), torch.from_numpy(mot_video_feats), \
                   torch.from_numpy(ques_words_idx), torch.from_numpy(label)

        else: # frameqa
            # No choices, only an index number of the answer in candidates
            label = np.array(ques['label_id'])
            return torch.from_numpy(region_feats), torch.from_numpy(region_boxes), \
                   torch.from_numpy(app_video_feats), torch.from_numpy(mot_video_feats), \
                   torch.from_numpy(ques_words_idx), torch.from_numpy(label)



if __name__ == '__main__':

    # Execute only once to subsample video frames up to 16 frames, and save the new hdf5 file in disk:
    old_file_path = '/data4/pengliang/HGA/dataset/tgif/features/TGIF_RESNET_pool5.hdf5'
    subsample_path = '/data4/pengliang/HGA/dataset/tgif/features/TGIF_RESNET_16.hdf5'
    subsample_frames(subsample_path, get_video_feats(old_file_path))


    # ques_types = ['action', 'count', 'transition', 'frameqa']
    #
    # train_set = QuestionDataset('Train')
    # train_set.create_vocab(min_cnt=2)
    # train_set.create_answers(min_cnt=2)
    #
    # # Tokenize and indexize:
    # QuestionDataset.tokenize(train_set.ques_total, train_set.vocab, train_set.ans2idx)
    # QuestionDataset.indexize(train_set.ques_total, train_set.word2idx, train_set.ans2idx)
    #
    # # Fix length and right padding:
    # padding_idx = train_set.vocab_size
    # train_set.fix_length(train_set.padding_idx, max_ques_len=13, max_ans_len=6)
    #
    # video_feats = get_video_feats('/data2/pengliang/subsample_frames.hdf5')
    # # Test every type of VideoQA dataset:
    # for type in ques_types:
    #     ques_set = train_set.ques_total[type]
    #     videoqa_set = VideoQA(type, video_feats, ques_set, train_set.vocab_size)
    #     print(type.upper() + ' videoqa length: ', len(videoqa_set))
    #     print(type.upper() + ' videoqa dataset[0]: \n', videoqa_set[0])
    #     print('## video feature sum: ', videoqa_set[0][0].sum())