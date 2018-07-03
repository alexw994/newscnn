import tensorflow as tf

weight_decay = 1e-4


def mapping(X, num_classes, embedding_size, is_train=False, reuse=False):
    with tf.variable_scope('classify', reuse=reuse):
        X = tf.squeeze(X, axis = -1)
        w = tf.get_variable(
            'kernel', [embedding_size, embedding_size],
            tf.float32,
            regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        biases = tf.get_variable('bias', [embedding_size], initializer=tf.constant_initializer(0.0))
        X = tf.nn.bias_add(tf.einsum('ijm,mk->ijk', X, w), biases)
        X = X[:, :, :, tf.newaxis]

        # X = conv2d(X, 64, 1, 1, 1, 1, name='conv')
        X = batch_norm(X, axis=3, train=is_train, name='bn')
        X = tanh(X)

        # k_h, k_w, d_h, d_w

        net1 = conv2d_block(X, 64, 3, embedding_size, 1, 1, name='conv1')
        net1 = tf.nn.max_pool(net1, ksize=[1, net1.get_shape()[1], 1, 1], strides=[1, 1, 1, 1], name='pool1', padding='VALID')
        net2 = conv2d_block(X, 64, 4, embedding_size, 1, 1, name='conv2')
        net2 = tf.nn.max_pool(net2, ksize=[1, net2.get_shape()[1], 1, 1], strides=[1, 1, 1, 1], name='pool2', padding='VALID')
        net3 = conv2d_block(X, 64, 5, embedding_size, 1, 1, name='conv3')
        net3 = tf.nn.max_pool(net3, ksize=[1, net3.get_shape()[1], 1, 1], strides=[1, 1, 1, 1], name='pool3', padding='VALID')

        net = tf.concat([net1, net2, net3], axis=3)

        net = dropout(flatten(net), is_train=is_train, name='dropout')
        net = _fc(net, num_classes, name = 'fc2')

        logits = net
        pred = tf.nn.softmax(logits)
    return logits, pred

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def tanh(x, name = 'tanh'):
    return  tf.nn.tanh(x, name)


def batch_norm(x, axis, momentum=0.9, epsilon=1e-5,train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                                         axis=axis,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         training=train,
                                         name=name)


def dropout(x, is_train, name):
    if is_train == True:
        return tf.nn.dropout(x, 0.6, name = name)
    else:
        return tf.identity(x)

def _fc(X, output_dim, name = 'fc', bias=True):
    with tf.variable_scope(name):
        in_channels = X.shape.as_list()[-1]

        w = tf.get_variable(
            'kernel', [in_channels, output_dim],
            tf.float32,
            regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        fc = tf.matmul(X, w)
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(X, w), biases)

        return fc

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.01, name='conv2d', bias=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv

def conv2d_block(input, out_dim, k_h, k_w, d_h, d_w, name, is_train = False):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k_h, k_w, d_h, d_w, name='conv2d')
        # net = batch_norm(net, train=is_train, name='bn')
        net = tanh(net)
        return net

def flatten(x):
    return tf.contrib.layers.flatten(x)


def global_avg(x):
    m, n= x.get_shape().as_list()[1:3]
    with tf.name_scope('global_avg'):
        net= tf.nn.avg_pool(x, ksize= [1, m, n, 1], strides = [1,1,1,1], padding = 'VALID')
        return net

