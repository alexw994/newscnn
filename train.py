import tensorflow as tf
from cnn import mapping
from get_data import *
import time
import os
import numpy as np

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
    num_samples = 156000
    num_valid = 36000
    embedding_size = 128
    batch_size = 32
    epoch = 401
    checkpoint_dir = _join_path('checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = 'NEWSCNN'

    # M_train, M_test, max_len = get_meta()
    # np.save(os.path.join(os.getcwd(), 'meta.npy'), [M_train, M_test, max_len])

    train_data = Seqdata(is_train=True, embedding_size=embedding_size)
    test_data = Seqdata(is_train=False, embedding_size=embedding_size)
    max_len = train_data.max_len

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [None, max_len, embedding_size, 1], name='input')
        labels = tf.placeholder(tf.int64, [None], name='label')

        logits, pred = mapping(inputs, num_classes=num_classes,
                               embedding_size=embedding_size, is_train=True)

        _, test_pred = mapping(inputs, num_classes=num_classes,
                               embedding_size=embedding_size, is_train=False, reuse=True)

        test_correct_pred = tf.equal(tf.argmax(test_pred, 1), tf.cast(labels, tf.int64))
        test_acc = tf.reduce_mean(tf.cast(test_correct_pred, tf.float32))

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_ = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = loss_ + l2_loss

        lr_decay = 0.98
        base_lr = tf.constant(3e-4)
        lr_decay_step = num_samples // batch_size * 2  # every epoch
        global_step = tf.placeholder(dtype=tf.float32, shape=())
        lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step,
                                        decay_rate= lr_decay)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(
                learning_rate=lr, beta1=1e-4).minimize(total_loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 1)
        step = 0
        could_load, step = _load(sess, saver, checkpoint_dir)

        max_steps = int(num_samples / batch_size * epoch)

        print('START TRAINING...')
        for _step in range(step + 1, max_steps+1):
            start_time=time.time()
            X_batch, Y_batch = train_data.next_batch(batch_size)

            feed_dict = {global_step:_step,
                         inputs: X_batch,
                         labels: Y_batch}
            _ = sess.run(train_op, feed_dict=feed_dict)
            if _step % 10 == 0:
                _loss, _acc = sess.run([total_loss, acc],
                                           feed_dict=feed_dict)
                print('global_step:{0}, remained_time:{1:.3f} min, acc:{2:.6f}, loss:{3:.6f}'.format
                      (_step, (max_steps - _step)*((time.time() - start_time))/60,  _acc, _loss))

            if _step % 3000 == 0:
                global_acc = 0
                for i in range(num_valid // batch_size + 1):
                    X_batch, Y_batch = test_data.next_batch(batch_size)

                    feed_dict = {global_step: _step,
                                 inputs: X_batch,
                                 labels: Y_batch}
                    _test_acc = sess.run(test_acc, feed_dict=feed_dict)
                    global_acc += _test_acc
                global_acc /= num_valid // batch_size + 1
                print('global_acc:{0:.6f}'.format(global_acc))

            if _step % 3000 == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=_step)
                print('Current model saved in ' + save_path)

        tf.train.write_graph(sess.graph_def, checkpoint_dir,model_name + '.pb')
        save_path = saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=max_steps)
        print('Final model saved in ' + save_path)
        sess.close()
        print('FINISHED TRAINING.')

if __name__ == '__main__':
    main()