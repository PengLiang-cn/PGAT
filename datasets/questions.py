import pandas as pd
import os
import pickle

"""
Custom class for the ques.csv file. As it receives the split name('train'/'test'),
we will get the question-dataset for all the 4 types of questions as below:
QuestionDataset.ques_total: {
        'action': ......
        'action_size': ......
        'count': ......
        'count_size': ......
        'transition': ......
        'transition_size': ......
        'frameqa': ......
        'frameqa_size': ......
        }
"""

class QuestionDataset(object):

    def __init__(self, split):

    ### MODIFY ###
        videos_missed = pickle.load(open('./data/pickle/videos_missed.pkl', 'rb'))
        videos_missed.append('tumblr_noqqk2knFU1tw52hco1_250') # buffer // element
        videos_missed.append('tumblr_nknf8nwBvU1rmklxfo1_250') # buffer // element

        assert split in ['Train', 'Test', 'Total']

        ques_total = {}
        ques_types = ['action', 'count', 'transition', 'frameqa']
        for type in ques_types:
            DIR_PATH = './data/questions'
            path = os.path.join(DIR_PATH, split + '_' + type + '_question.csv')
            ques_df = pd.read_csv(path, sep='\t', encoding = 'utf-8', header = None, engine = 'python')
            # if type == 'action':
            #     print('\n######## First 5 row: ########\n', ques_df.head())

            ques_size, _ = ques_df.shape
            if type in ['action', 'transition']:
                number_videos = 0
                sub_ques = []
                for i in range(1, ques_size):
                    ## IMPORTANT ##
                    video_name = ques_df[0][i]
                    if video_name in videos_missed:
                        number_videos = number_videos + 1
                        continue

                    columns = ['gif_name', 'question', 'a1', 'a2', 'a3', 'a4', 'a5',
                               'answer', 'type', 'vid_id']
                    ques = {key: ques_df[col][i] for col, key in enumerate(columns)}

                    str = ques_df[1][i].lower().replace('?', '').replace('.', '').replace(',', '')
                    ques['ques_words'] = str.split() # easy to create vocab
                    sub_ques.append(ques)
                if type == 'action':
                   ques_total['action'] = sub_ques
                   ques_total['action_size'] = len(sub_ques)
                else:
                    ques_total['transition'] = sub_ques
                    ques_total['transition_size'] = len(sub_ques)
                print('%s, %s miss video: %d'%(type, split, number_videos))
            elif type == 'frameqa':
                number_videos = 0
                sub_ques = []
                for i in range(1, ques_size):
                    ## IMPORTANT ##
                    video_name = ques_df[0][i]
                    if video_name in videos_missed:
                        number_videos = number_videos + 1
                        continue

                    columns = ['gif_name', 'question', 'answer', 'QA_type', 'type',
                               'vid_id', 'description']
                    ques = {key: ques_df[col][i] for col, key in enumerate(columns)}
                    str = ques_df[1][i].lower().replace('?', '').replace('.', '').replace(',', '')

                    ques['ques_words'] = str.split() # easy to create vocab
                    sub_ques.append(ques)
                ques_total['frameqa'] = sub_ques
                ques_total['frameqa_size'] = len(sub_ques)
                print('%s, %s miss video: %d'%(type, split, number_videos))
            else:
                number_videos = 0
                sub_ques = []
                for i in range(1, ques_size):
                    ## IMPORTANT ##
                    video_name = ques_df[0][i]
                    if video_name in videos_missed:
                        number_videos = number_videos + 1
                        continue

                    columns = ['gif_name', 'question', 'answer', 'type', 'vid_id']
                    ques = {key: ques_df[col][i] for col, key in enumerate(columns)}

                    str = ques_df[1][i].lower().replace('?', '').replace('.', '').replace(',', '')
                    ques['ques_words'] = str.split() # easy to create vocab
                    ques['answer'] = max(int(ques['answer']), 1)
                    sub_ques.append(ques)
                ques_total['count'] = sub_ques
                ques_total['count_size'] = len(sub_ques)
                print('%s, %s miss video: %d'%(type, split, number_videos))

        # It contains four types of ques at the same time:
        self.ques_total = ques_total

    # Only for training set
    def create_vocab(self, min_cnt=1):
        print('Create vocab start...')
        word_cnt = {}
        # We need to collect all the words occurred in 4 types of training questions:
        ques_types = ['action', 'count', 'transition', 'frameqa']
        for type in ques_types:
            for ques in self.ques_total[type]:
                for word in ques['ques_words']:
                    word_cnt[word] = word_cnt.get(word, 0) + 1
        self.vocab = [w for w in list(word_cnt.keys()) if word_cnt[w] >= min_cnt]
        self.vocab.append('UNK')
        self.vocab_size = len(self.vocab)
        self.padding_idx = len(self.vocab) # index for zero-padding

        # Use the unique index to replace every word:
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}
        print('Create vocab finished...')


    # Only for FrameQA
    def create_answers(self, min_cnt=1):
        """
        In FrameQA, we need to answer the question without given choices, which is similar
        to VisualQA. So we collect all the words occurred in the labels to create candidates
        for answers. Finally, FrameQA can be regarded as a multi-class task.
        """
        print('Create answers start...')
        ans_cnt = {}
        for ques in self.ques_total['frameqa']:
            ans = ques['answer']
            ans_cnt[ans] = ans_cnt.get(ans, 0) + 1
        # Remain the ans occurred more than 'min_cnt' times only:
        self.ans_set = [ans for ans in list(ans_cnt.keys()) if ans_cnt[ans] >= min_cnt]
        self.ans_set.append('UNK')
        self.ans_size = len(self.ans_set)

        # use the unique index to replace every ans:
        self.ans2idx = {ans: i for i, ans in enumerate(self.ans_set)}
        self.idx2ans = {i: ans for i, ans in enumerate(self.ans_set)}
        print('Create answers finished...')

    @staticmethod
    def tokenize(ques_total, vocab, ans2id):
        print('Tokenize start...')
        ques_types = ['action', 'count', 'transition', 'frameqa']
        for type in ques_types:
            for ques in ques_total[type]:
                # Replace the words missed in vocab as 'UNK'
                ques['ques_words_UNK'] = [w if w in vocab else 'UNK' for w in ques['ques_words']]

                # Tokenize the 5 given choices as well:
                if type in ['action', 'transition']:
                    for i in range(1, 6):
                        name = 'a' + str(i); name_UNK = name + '_UNK'
                        ques[name_UNK] = [w if w in vocab else 'UNK' for w in ques[name].split()]

                # Tokenize the label(a word) only for frameqa:
                if type == 'frameqa':
                    if ques['answer'] not in ans2id:
                        ques['label_UNK'] = 'UNK'
                        # label_id = ans2id['UNK']
                    else:
                        ques['label_UNK'] = ques['answer']
                        # label_id = ans2id[ques['answer']]
                    # ques['label_id'] = label_id

        print('Tokenize finished...')

    @staticmethod
    def indexize(ques_total, word2idx, ans2id):
        print('Indexize start...')
        # Encode the question from word sequence to index sequence:
        ques_types = ['action', 'count', 'transition', 'frameqa']
        for type in ques_types:
            for ques in ques_total[type]:
                ques['words_idx_UNK'] = [word2idx[w] for w in ques['ques_words_UNK']]

                # Encode the 5 given answers as well:
                if type in ['action', 'transition']:
                    for i in range(1, 6):
                        name = 'a' + str(i) + '_UNK'; idx_name = name + '_idx'
                        ques[idx_name] = [word2idx[w] for w in ques[name]]

                # Indexize the label(a word) only for frameqa:
                if type == 'frameqa':
                    label_UNK = ques['label_UNK']
                    ques['label_id'] = ans2id[label_UNK]

        print('Indexize finished...')

    # Convert the questions and answers into a fixed size by right padding.
    def fix_length(self, padding_idx, max_ques_len=13, max_ans_len=6):
        print('Fix length start...')
        # Fix the questions:
        for type in ['action', 'count', 'transition', 'frameqa']:
            for ques in self.ques_total[type]:
                ques_len = len(ques['ques_words'])
                if ques_len >= max_ques_len:
                    ques['words_idx_UNK'] = ques['words_idx_UNK'][: max_ques_len]
                else:
                    # Right padding. Here, both '+' and '*' are operations for python list.
                    ques['words_idx_UNK'] = ques['words_idx_UNK'] + (max_ques_len - ques_len) * [padding_idx]

        # Fix the 5 given choices:
        for type in ['action', 'transition']:
            for ques in self.ques_total[type]:
                for i in range(1, 6):
                    ans_name = 'a' + str(i) + '_UNK_idx'
                    ans_len = len(ques[ans_name])
                    if ans_len >= max_ans_len:
                        ques[ans_name] = ques[ans_name][: max_ans_len]
                    else:
                        # Right padding.
                        ques[ans_name] = ques[ans_name] + (max_ans_len - ans_len) * [padding_idx]
        print('Fix length finished...')


def dump_to_file(dir_path, file_name, obj):
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(file_path):
        pickle.dump(obj, open(file_path, 'wb'))
        print('File dumped success!')


if __name__ == '__main__':

    DIR_PKL = '/home/yangsj/GCN_VideoQA/data/pickle'
    train_set = QuestionDataset('Train')
    ques_types = ['action', 'count', 'transition', 'frameqa']

    # test vocab:
    train_set.create_vocab(min_cnt=2)
    print('Vocab size: ', train_set.vocab_size)

    dump_to_file(DIR_PKL, 'vocab3.pkl', train_set.vocab)
    dump_to_file(DIR_PKL, 'word2idx3.pkl', train_set.word2idx)
    dump_to_file(DIR_PKL, 'idx2word3.pkl', train_set.idx2word)

    # test candidate answers:
    train_set.create_answers(min_cnt=2)
    print('Answer set size: ', train_set.ans_size)
    dump_to_file(DIR_PKL, 'ans_set3.pkl', train_set.ans_set)
    dump_to_file(DIR_PKL, 'ans2idx3.pkl', train_set.ans2idx)
    dump_to_file(DIR_PKL, 'idx2ans3.pkl', train_set.idx2ans)

    # Tokenize and indexize:
    QuestionDataset.tokenize(train_set.ques_total, train_set.vocab, train_set.ans2idx)
    QuestionDataset.indexize(train_set.ques_total, train_set.word2idx, train_set.ans2idx)

    # Fix length and right padding:
    padding_idx = train_set.vocab_size
    train_set.fix_length(train_set.padding_idx, max_ques_len=13, max_ans_len=6)
    for type in ques_types:
        print('Ques type: {}, Length: {}'.format(type, len(train_set.ques_total[type])))
        print(train_set.ques_total[type][0])

