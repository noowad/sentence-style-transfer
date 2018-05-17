import tensorflow as tf
from data import load_sentences, create_data, load_glove, load_vocab
from hyperparams import Hyperparams as hp
from graph import DeleteOnlyGraph, DeleteAndRetrieveGraph
import argparse
import numpy as np
import os
import codecs


def inference(mode):
    # data load
    neg_lines = load_sentences(hp.data_path + '/delete/delete.test.0', False)
    pos_lines = load_sentences(hp.data_path + '/delete/delete.test.1', False)
    word_w2i, word_i2w = load_vocab()
    # mode
    if mode == 'delete_only':
        # assert(DeleteOnlyGraph(X, Y, attribute_labels))
        g = DeleteOnlyGraph()
    else:
        g = DeleteAndRetrieveGraph()
    with g.graph.as_default(), tf.Session() as sess:
        sv = tf.train.Saver()
        # Restore parameters
        print("Parameter Restoring...")
        sv.restore(sess, './logdir/' + mode + '/model.ckpt')

        # Inference
        if not os.path.exists(hp.data_path + '/generate'): os.mkdir(hp.data_path + '/generate')
        for num in ['0', '1']:
            if num == '0':
                X, Y, A, Attribute_labels, Sources, Targets, Attributes = create_data(neg_lines, 1, mode='inference')
            else:
                X, Y, A, Attribute_labels, Sources, Targets, Attributes = create_data(pos_lines, 0, mode='inference')
            with codecs.open(hp.data_path + '/generate/' + mode + '.test.' + num, "w", "utf-8") as fout:
                for i in range(len(X)):
                    x = X[i: i + 1]
                    attri_label = np.array(Attribute_labels[i:i + 1]).reshape((-1, 1))
                    a = A[i:i + 1]
                    sources = Sources[i: i + 1]
                    targets = Targets[i: i + 1]

                    preds = np.zeros((1, hp.max_len), np.int32)
                    for j in range(hp.max_len):
                        if mode == 'delete_only':
                            _preds = sess.run(g.pred, {g.x: x, g.y: preds, g.attributes: attri_label})
                        else:
                            _preds = sess.run(g.pred, {g.x: x, g.y: preds, g.a: a})
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = " ".join(word_i2w[idx] for idx in pred).split("<END>")[0].strip()
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
