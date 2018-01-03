"""MNIST with capsule network."""

import numpy as np
import os
import tensorflow as tf
import capsnet

from config import cfg
from tqdm import tqdm
from utils import load_data, shuffled_batch_queue


def save_to():
    """Save accuracy, loss of training or evaluation as csv format."""
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train():
    """Train the capsule network."""
    # Load train and validation data.
    # Use one hot as ground truth for train, use labels as is for validation.
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    X, labels = shuffled_batch_queue(trX, trY, cfg.batch_size, cfg.num_threads)
    Y = tf.one_hot(labels, depth=10, axis=1, dtype=tf.float32)

    # Build graph
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model = capsnet.model(X)
    v_length, prediction = capsnet.predict(model)
    decoded = capsnet.decoder(model, prediction)
    margin_loss, reconstruction_loss, total_loss = capsnet.loss(X, Y, v_length, decoded)
    train_op = capsnet.train_op(total_loss, global_step)
    train_summary = capsnet.summary(decoded, margin_loss, reconstruction_loss, total_loss)
    accuracy = capsnet.accuracy(labels, prediction)

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # XXX What is this for?
    # start training or resume training from last checkpoint
    supervisor = tf.train.Supervisor(logdir=cfg.logdir, save_model_secs=0)
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                # Train input: X <- trX, Y <- one hot trY
                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([train_op, total_loss, accuracy, train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(train_op)

                # Validation input: X <- valX,  Y <- valY (labels as is)
                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(accuracy, {X: valX[start:end], labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            # checkpoint
            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation():
    """Evaluate / test the accuracy of trained model."""
    # teY are labels
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
    labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))

    # Build graph
    model = capsnet.model(X)
    _, prediction = capsnet.predict(model)
    accuracy = capsnet.accuracy(labels, prediction)

    fd_test_acc = save_to()
    # start training or resume training from last checkpoint
    supervisor = tf.train.Supervisor(logdir=cfg.logdir, save_model_secs=0)
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(accuracy, {X: teX[start:end], labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_accuracy.txt')


def main(_):
    """Run train or evaluation."""
    graph = tf.Graph()
    with graph.as_default():
        if cfg.is_training:
            train()
        else:
            evaluation()


if __name__ == "__main__":
    tf.app.run()
