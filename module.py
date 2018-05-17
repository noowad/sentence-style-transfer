import tensorflow as tf
from data import load_glove
from hyperparams import Hyperparams as hp


def embed(vocab_size, num_units, scope="embedding", reuse=None, trainable=False):
    '''
    Embedding Lookup-table.
    '''
    with tf.variable_scope(scope, initializer=tf.random_uniform_initializer(-0.1, 0.1), reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       trainable=trainable)
    return lookup_table


def gru(inputs, num_units=hp.gru_size, initial_state=None, bidirection=False, scope="gru", reuse=None):
    with tf.variable_scope(scope, initializer=tf.random_uniform_initializer(-0.1, 0.1), reuse=reuse):
        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2), tf.concat(final_state, 1)
        else:
            outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
            return outputs, final_state


def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    with tf.variable_scope(scope, initializer=tf.random_uniform_initializer(-0.1, 0.1), reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                                   memory,
                                                                   normalize=True,
                                                                   probability_fn=tf.nn.softmax)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, num_units)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32)  # ( N, T', 16)
    return outputs
