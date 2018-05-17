from hyperparams import Hyperparams as hp
import tensorflow as tf
from module import *


def encode(encoder_inputs, attributes, scope="encoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if hp.neural_mode == 'delete_only':
            _, final_state = gru(encoder_inputs, hp.gru_size // 2, bidirection=True)  # (N, T, E)
            enc = tf.add(final_state, attributes)
        else:
            _, contents_final_state = gru(encoder_inputs, hp.gru_size // 4, bidirection=True)  # (N, T, E)
            _, attri_final_state = gru(attributes, hp.gru_size // 2, bidirection=False)  # (N, T, E)
            # enc = tf.add(contents_final_state, attri_final_state)
            enc = tf.concat([contents_final_state, attri_final_state], axis=1)
    return enc


def decode(decoder_inputs, encoder_outputs, out_dim, scope="decoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Attention RNN
        # dec = attention_decoder(decoder_inputs, memory, hp.embed_size)  # (N, T', E)

        # Decoder RNNs
        dec, _ = gru(decoder_inputs, hp.gru_size, encoder_outputs, False, scope="decoder_gru")  # (N, T', E)

        outputs = tf.layers.dense(dec, out_dim,
                                  kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                  bias_initializer=tf.random_uniform_initializer(-0.1, 0.1))

    return outputs
