"""Mostly copied from naturomics/CapsNet-Tensorflow.

Modifications are;
(1) Capsules in same channel ***share*** a W_ij in routing.
(2) Function instead of class.
"""
import tensorflow as tf
from config import cfg
epsilon = 1e-9


#
# Capsule layers
#


def squash(v):
    """Return squashed v."""
    square_norm = tf.reduce_sum(tf.square(v), -2, keep_dims=True)
    return square_norm / (1 + square_norm) * v / tf.sqrt(square_norm + epsilon)


def conv_caps(input, num_outputs, kernel_size, stride, vec_len):
    """Return PrimaryCaps layer, convolutional capsule layer."""
    caps = tf.contrib.layers.conv2d(input, num_outputs * vec_len,
                                    kernel_size, stride, padding="VALID",
                                    activation_fn=tf.nn.relu)
    caps = tf.reshape(caps, [cfg.batch_size, -1, vec_len, 1])
    caps = squash(caps)
    return caps


def fc_caps(input, num_outputs, vec_len):
    """Return DigitCaps layer, fully connected layer."""
    with tf.variable_scope('routing'):
        uh = conv_to_fc(input)
        caps = routing(uh, num_outputs)
        caps = tf.squeeze(caps, axis=1)
        return caps


# Graph computing *u_hat_IJ* of Equation. 2 in the paper.
def conv_to_fc(u):
    """Return FC-wise contribution from conv capsules to digit capsules."""
    # Reshape, tile and transpose the u for tf.scan W * u (send 36 outer)
    #                                                  TTTTTT  TT                            TT  TTTTTT
    # u: =(reshape)=> [bs, 32, 36, 1, 8, 1] =(tile)=> [bs, 32, 36, 10, 8, 1] =(transpose)=> [36, bs, 32, 10, 8, 1]
    #                      RRRRRR  t                               tt
    u = tf.reshape(u, [cfg.batch_size, 32, -1, 1, 8, 1])
    u = tf.tile(u, [1, 1, 1, 10, 1, 1])
    u = tf.transpose(u, perm=[2, 0, 1, 3, 4, 5])
    assert u.get_shape() == [36, cfg.batch_size, 32, 10, 8, 1]

    # W: [bs, 32, 10, 8, 16], multiplying 36 times by tf.scan
    W = tf.get_variable('Weight', shape=[1, 32, 10, 8, 16], dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    # XXX Is this tile avoidable also (by tf.scan)?
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])

    # Eq.2, u_hat
    # [..., 8, 16].T x [..., 8, 1] => [..., 16, 1] => [36, batch_size, 32, 10, 16, 1]
    uh = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), u,
                 initializer=tf.zeros([cfg.batch_size, 32, 10, 16, 1]))
    assert uh.get_shape() == [36, cfg.batch_size, 32, 10, 16, 1]

    # Transpose and reshape uh back for sum_I {c (*) uh}
    #  TT  TTTTTT                             TTTTTT  TT
    # [36, bs, 32, 10, 16, 1] =(transpose)=> [bs, 32, 36, 10, 16, 1] =(reshape) => [bs, 1152, 10, 16, 1]
    #                                             RRRRRR                                RRRR
    uh = tf.transpose(uh, perm=[1, 2, 0, 3, 4, 5])
    uh = tf.reshape(uh, shape=[cfg.batch_size, -1, 10, 16, 1])
    assert uh.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]
    return uh


def routing(uh, num_outputs):
    """Route by dynamic agreement."""
    # In forward (inner iterations), uh_stopped = uh.
    # In backward, no gradient passed back from uh_stopped to uh.
    uh_stopped = tf.stop_gradient(uh, name='stop_gradient')
    b = tf.zeros([cfg.batch_size, uh.shape[1].value, num_outputs, 1, 1])  # b: [bs, 1152, 10, 1, 1]
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c = tf.nn.softmax(b, dim=2)  # [bs, 1152, 10, 1, 1]
            # At last iteration, use `uh` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # weighting uh with c, element-wise in the last two dims
                s = tf.reduce_sum(tf.multiply(c, uh), axis=1, keep_dims=True)
                v = squash(s)
                assert v.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:
                # Inner iterations, do not apply backpropagation
                s = tf.reduce_sum(tf.multiply(c, uh_stopped), axis=1, keep_dims=True)
                v = squash(s)
                # tile from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # for matmul, in the last two dim: [16, 1].T x [16, 1] => [1, 1]
                v_tiled = tf.tile(v, [1, 1152, 1, 1, 1])
                uh_produce_v = tf.matmul(uh_stopped, v_tiled, transpose_a=True)  # Agreement
                assert uh_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]
                b += uh_produce_v
    return(v)


#
# Network architecture
#


def model(X):
    """Return capsule network for MNIST."""
    with tf.variable_scope('Conv1_layer'):
        conv1 = tf.contrib.layers.conv2d(X, num_outputs=256,
                                         kernel_size=9, stride=1,
                                         padding='VALID')
        assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]
    with tf.variable_scope('PrimaryCaps'):
        caps1 = conv_caps(conv1, num_outputs=32, kernel_size=9, stride=2, vec_len=8)
    with tf.variable_scope('DigitCaps'):
        caps2 = fc_caps(caps1, num_outputs=10, vec_len=16)
    return caps2


def predict(caps2):
    """Return prediction with argmax."""
    with tf.variable_scope('Prediction'):
        # softmax(|v|), where v: [bs, 10, 16, 1]
        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)
        assert v_length.get_shape() == [cfg.batch_size, 10, 1, 1]
        softmax_v = tf.nn.softmax(v_length, dim=1)
        # index of max softmax val among the 10 digit
        prediction = tf.to_int32(tf.argmax(softmax_v, axis=1))
        assert prediction.get_shape() == [cfg.batch_size, 1, 1]
        prediction = tf.reshape(prediction, shape=(cfg.batch_size, ))
        return v_length, prediction


def decoder(caps2, prediction):
    """Return decoder for reconstruction of image."""
    # Masking
    with tf.variable_scope('Masking'):
        # batch size of predictions (labels)
        candid = []
        for index in range(cfg.batch_size):
            v = caps2[index][prediction[index], :]  # [16, 1]
            candid.append(tf.reshape(v, shape=(1, 1, 16, 1)))
        candid = tf.concat(candid, axis=0)
        assert candid.get_shape() == [cfg.batch_size, 1, 16, 1]

    # Reconstruct batch size of images with 3 FC layers
    with tf.variable_scope('Decoder'):
        v = tf.reshape(candid, shape=(cfg.batch_size, -1))  # [bs, 1, 16, 1] => [bs, 16]
        fc1 = tf.contrib.layers.fully_connected(v, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        assert fc2.get_shape() == [cfg.batch_size, 1024]
        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)
        return decoded


def loss(X, Y, v_length, decoded):
    """Return loss."""
    # These work by virtue of broadcasting (0, m_plus, m_minus),
    # max_l = max(0, m_plus-||v_k||)^2
    # max_r = max(0, ||v_k||-m_minus)^2
    # v_length: [bs, 10, 1, 1]
    max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
    assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]
    max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
    assert max_r.get_shape() == [cfg.batch_size, 10, 1, 1]
    # reshape: [bs, 10, 1, 1] => [bs, 10]
    max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
    max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

    # 1. Margin loss (T == Y)
    L = Y * max_l + cfg.lambda_val * (1 - Y) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))

    # 2. Reconstruction loss
    orgin = tf.reshape(X, shape=(cfg.batch_size, -1))
    squared = tf.square(decoded - orgin)
    reconstruction_loss = tf.reduce_mean(squared)

    # 3. Total loss
    # The paper uses sum of squared error as reconstruction error, but we have
    # used reduce_mean to calculate MSE.  In order to keep in line with the
    # paper, the regularization scale should be 0.0005*784=0.392
    total_loss = margin_loss + cfg.regularization_scale * reconstruction_loss

    return margin_loss, reconstruction_loss, total_loss


def train_op(total_loss, global_step):
    """Return train operation."""
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(total_loss, global_step=global_step)


def summary(decoded, margin_loss, reconstruction_loss, total_loss):
    """Return train summary."""
    train_summary = []
    train_summary.append(tf.summary.scalar('train/margin_loss', margin_loss))
    train_summary.append(tf.summary.scalar('train/reconstruction_loss', reconstruction_loss))
    train_summary.append(tf.summary.scalar('train/total_loss', total_loss))
    recon_img = tf.reshape(decoded, shape=(cfg.batch_size, 28, 28, 1))
    train_summary.append(tf.summary.image('reconstruction_img', recon_img))
    train_summary = tf.summary.merge(train_summary)
    return train_summary


def accuracy(labels, prediction):
    """Return accuracy."""
    correct_prediction = tf.equal(tf.to_int32(labels), prediction)
    return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
