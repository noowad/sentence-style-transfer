import tensorflow as tf
import os
from tqdm import tqdm
import random
from delete import delete
from data import load_sentences, create_data, load_glove
from hyperparams import Hyperparams as hp
from graph import DeleteOnlyGraph, DeleteAndRetrieveGraph
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', type=str, default='delete_only',
                        help='Enter train mode')
    par_args = parser.parse_args()
    hp.neural_mode = par_args.mode
    if not os.path.exists('logdir/' + hp.neural_mode): os.makedirs('logdir/' + hp.neural_mode)

    # data load
    neg_lines = load_sentences(hp.data_path + '/delete/delete.train.0', False)
    pos_lines = load_sentences(hp.data_path + '/delete/delete.train.1', False)
    valid_neg_lines = load_sentences(hp.data_path + '/delete/delete.dev.0', False)
    valid_pos_lines = load_sentences(hp.data_path + '/delete/delete.dev.1', False)

    print('Creating datas...')
    neg_X, neg_Y, neg_A, neg_attribute_labels, _, _, _ = create_data(neg_lines, 0)
    pos_X, pos_Y, pos_A, pos_attribute_labels, _, _, _ = create_data(pos_lines, 1)
    X = np.concatenate([neg_X, pos_X], axis=0)
    Y = np.concatenate([neg_Y, pos_Y], axis=0)
    A = np.concatenate([neg_A, pos_A], axis=0)
    a_labels = np.array(neg_attribute_labels + pos_attribute_labels).reshape((-1, 1))

    valid_neg_X, valid_neg_Y, valid_neg_A, valid_neg_attribute_labels, _, _, _ = create_data(valid_neg_lines, 0)
    valid_pos_X, valid_pos_Y, valid_pos_A, valid_pos_attribute_labels, _, _, _ = create_data(valid_pos_lines, 1)
    valid_X = np.concatenate([valid_neg_X, valid_pos_X], axis=0)
    valid_Y = np.concatenate([valid_neg_Y, valid_pos_Y], axis=0)
    valid_A = np.concatenate([valid_neg_A, valid_pos_A], axis=0)
    valid_a_labels = np.array(valid_neg_attribute_labels + valid_pos_attribute_labels).reshape((-1, 1))
    print('Loading glove matrix...')
    weights = load_glove()
    # mode
    if hp.neural_mode == 'delete_only':
        g = DeleteOnlyGraph()
    else:
        g = DeleteAndRetrieveGraph()

    data_size = X.shape[0]
    data_list = list(range(data_size))
    with g.graph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Initialize
            sess.run(tf.global_variables_initializer())
            # pre-trained GloVe assign
            if hp.is_glove:
                sess.run(g.glove_assign, {g.glove_weights: weights})
            best_valid_loss = 100000.
            for epoch in range(1, hp.num_epochs):
                np.random.shuffle(data_list)
                # Train
                train_loss = 0.
                num_batch = data_size / hp.batch_size
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    word_ids = data_list[step * hp.batch_size:step * hp.batch_size + hp.batch_size]
                    if hp.neural_mode == 'delete_only':
                        _, t_loss, gs = sess.run([g.train_op, g.mean_loss, g.global_step], {g.x: X[word_ids],
                                                                                            g.y: Y[word_ids],
                                                                                            g.attributes:
                                                                                                a_labels[word_ids]})
                    else:
                        _, t_loss, gs = sess.run([g.train_op, g.mean_loss, g.global_step], {g.x: X[word_ids],
                                                                                            g.y: Y[word_ids],
                                                                                            g.a: A[word_ids]})
                    train_loss += t_loss
                    if step % (num_batch / 10) == 0:
                        print('\tstep:{} train_loss:{}'.format(gs, t_loss))
                train_loss /= num_batch
                # Validation
                valid_loss = 0.
                for idx in range(0, len(valid_X), hp.batch_size):
                    word_ids = list(range(len(valid_X)))[idx:idx + hp.batch_size]
                    if hp.neural_mode == 'delete_only':
                        v_loss = sess.run(g.mean_loss, {g.x: valid_X[word_ids],
                                                        g.y: valid_Y[word_ids],
                                                        g.attributes: valid_a_labels[word_ids]})
                    else:
                        v_loss = sess.run(g.mean_loss, {g.x: valid_X[word_ids],
                                                        g.y: valid_Y[word_ids],
                                                        g.a: valid_A[word_ids]})
                    valid_loss += v_loss
                valid_loss /= (len(valid_X) / hp.batch_size)
                print("[epoch{}] train_loss={:.2f} validate_loss={:.2f} ".format(epoch, train_loss, valid_loss))
                # Stopping
                if valid_loss <= best_valid_loss * 0.999:
                    best_valid_loss = valid_loss
                    saver.save(sess, 'logdir/' + hp.neural_mode + '/model.ckpt')
                else:
                    if hp.is_earlystopping:
                        print("Early Stopping...")
                        break
