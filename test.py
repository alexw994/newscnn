import tensorflow as tf
from cnn import mapping
from get_data import *
from gensim.models import Word2Vec

import numpy as np
import time
import os

def _load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0

def _join_path(path, *paths):
    if not paths:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    else:
        rst = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        for i in paths:
            rst = os.path.join(rst, i)
        return rst

def main():
    num_classes = 18
    embedding_size = 128
    batch_size = 64

    checkpoint_dir = os.path.join(os.path.abspath(os.getcwd()), 'checkpoint')

    # M_train, M_test, max_len = get_meta()
    # np.save('meta.npy', [M_train, M_test, max_len])

    M_train, M_test, max_len = np.load(_join_path('meta.npy'))[()]
    label = get_label()
    model = Word2Vec.load(_join_path('w2v_model', 'word2vec.model'))
    # vocab = get_vocab()

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [None, max_len, embedding_size, 1], name='input')
        labels = tf.placeholder(tf.int64, [None], name='label')


        logits, pred = mapping(inputs, num_classes=num_classes,
                               embedding_size=embedding_size, is_train=False)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 3)
        step = 0
        could_load, step = _load(sess, saver, checkpoint_dir)

        nb_testing = M_test.shape[0]
        y_pred = np.zeros((nb_testing//batch_size + 1) * batch_size)

        for i in range(nb_testing//batch_size + 1):
            seq_batch = M_test.seq[i * batch_size  : (i + 1) * batch_size]

            X_batch = np.zeros(shape=[batch_size, max_len, embedding_size])
            for n1, seq in enumerate(seq_batch):
                for n2, word in enumerate(seq):
                    try:
                        vec = np.array(model[word], dtype=np.float32)[np.newaxis, :]
                    except KeyError:
                        vec = np.zeros(shape=[1, embedding_size], dtype=np.float32)
                    X_batch[n1, n2,  :] = vec

            X_batch = X_batch[:, :, :, np.newaxis]

            y_pred[i * batch_size  : (i + 1) * batch_size] = sess.run(tf.argmax(pred, 1), feed_dict={inputs: X_batch})

            print(i,'/',nb_testing//batch_size + 1)

        acc_matrix = np.zeros(shape = (num_classes, num_classes))
        for i in range(nb_testing):
            acc_matrix[int(y_pred[i]), int(label.index(M_test['class_number'][i]))] +=1

        np.savetxt('acc.txt', acc_matrix)
        acc = np.sum(np.diag(acc_matrix))/ nb_testing
        print(acc)

if __name__ == '__main__':
    main()