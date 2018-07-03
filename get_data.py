import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
def _join_path(path, *paths):
    if not paths:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    else:
        rst = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        for i in paths:
            rst = os.path.join(rst, i)
        return rst

def get_label():
    label = []
    with open(_join_path('txt_data','id2tag.txt')) as f:
        for l in f:
            label.append(l.split('\t')[0])
    return label

class Seqdata():
    def __init__(self, embedding_size, is_train):
        self._M_train, self._M_test, self._max_len = np.load(os.path.join(os.getcwd(), 'meta.npy'))[()]
        self._label = self.get_label()
        self._model = Word2Vec.load(_join_path('w2v_model', 'word2vec.model'))
        self._vocab = self.get_vocab()
        self._index_in_epoch = 0
        self._embedding_size = embedding_size
        if is_train:
            self._data = self._M_train
        else:
            self._data = self._M_test
        self._num_examples = len(self._data)
        self._epochs_completed = 0

    def get_label(self):
        label = []
        with open(_join_path('txt_data','id2tag.txt')) as f:
            for l in f:
                label.append(l.split('\t')[0])
        return label

    def get_vocab(self):
        vocab_list = []
        with open(_join_path('vocab.txt'), 'r', encoding='utf-8') as f:
            for l in f:
                if int(l.split('\t')[1].split('\n')[0]) > 2:
                    vocab_list.append(l.split('\t')[0])
        return list(set(vocab_list))


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            self._data = self._data.sample(frac=1).reset_index(drop=True)
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        M_batch = self._data[start : end]
        X_batch, Y_batch = self.seq2batch(M_batch, batch_size)

        return X_batch, Y_batch

    def seq2batch(self, M_batch, batch_size):
        X_batch = np.zeros(shape=[batch_size, self._max_len, self._embedding_size])
        Y_batch = np.zeros(shape=[batch_size])

        for m, row in enumerate(M_batch.iterrows()):
            seq = row[1].seq
            class_n = np.array([self._label.index(row[1].class_number)], np.int64)
            X_array = np.zeros(shape=[len(seq), self._embedding_size])
            for n, word in enumerate(seq):
                try:
                    vec = np.array(self._model[word], dtype=np.float32)[np.newaxis, :]
                except KeyError:
                    vec = np.zeros(shape=[1, self._embedding_size], dtype=np.float32)

                X_array[n, :] = vec

            X_batch[m, 0:X_array.shape[0], :] = X_array
            Y_batch[m] = class_n

        X_batch = X_batch[:, :, :, np.newaxis]
        return X_batch, Y_batch


    @property
    def M_train(self):
        return self._M_train

    @property
    def M_test(self):
        return self._M_test

    @property
    def max_len(self):
        return self._max_len

    @property
    def label(self):
        return self._label

    @property
    def model(self):
        return self._model

    @property
    def vocab(self):
        return self._vocab


def get_meta():

    meta_train_data = []
    with open(_join_path('train_cut.txt'), 'r', encoding='utf-8') as f:
        for l in f:
            label = l.split('\t')[0]
            data = l.split('\t')[1].split('\n')[0].split(' ')
            meta_train_data.append([label, data])
    M_train = pd.DataFrame(meta_train_data)
    M_train.columns = ['class_number', 'seq']
    M_train = M_train.sample(frac=1).reset_index(drop=True)

    meta_test_data = []
    with open(_join_path('test_cut.txt'), 'r', encoding='utf-8') as f:
        for l in f:
            label = l.split('\t')[0]
            data = l.split('\t')[1].split('\n')[0].split(' ')
            meta_test_data.append([label, data])
    M_test = pd.DataFrame(meta_test_data)
    M_test.columns = ['class_number', 'seq']
    M_test = M_test.sample(frac=1).reset_index(drop=True)

    max_len = np.max([len(row[1].seq) for row in M_train.iterrows()] +
                     [len(row[1].seq) for row in M_test.iterrows()])

    return M_train, M_test, max_len

def get_batch2(vocab, batch_size = None, M_train = None, M_test= None, label= None, max_len = None, is_train = True):
    if is_train == True:
        M_batch = M_train.sample(32)
    else:
        M_batch = M_test.sample(32)

    X_batch = np.zeros(shape=[batch_size, max_len])
    Y_batch = np.zeros(shape=[batch_size])

    for m, row in enumerate(M_batch.iterrows()):
        seq = row[1].seq
        class_n = np.array([label.index(row[1].class_number)], np.int64)
        X_array = np.zeros(shape=[max_len])

        for n, word in enumerate(seq):
            try:
                X_array[n] = int(vocab.index(word))
            except ValueError:
                pass

        X_batch[m, :] = X_array
        Y_batch[m] = class_n
    X_batch = np.array(X_batch, np.int32)

    return X_batch, Y_batch

import jieba
def process(str, model):
    str = jieba.cut(str, cut_all=False)
    X_batch = np.zeros(shape=[1, 25, 64])

    for n, word in enumerate(str):
        try:
            vec = np.array(model[word], dtype=np.float32)[np.newaxis, :]
        except KeyError:
            vec = np.zeros(shape=[1, 64], dtype=np.float32)

        X_batch[0, n, :] = vec
    return X_batch
