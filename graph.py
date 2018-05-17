import tensorflow as tf
from module import *
from network import *
from utils import *
from data import load_vocab


class DeleteOnlyGraph():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.attributes = tf.placeholder(tf.int32, shape=(None, 1))

            # Vocab
            self.word_w2i, _ = load_vocab()

            # Embedding table
            self.glove_weights = tf.placeholder(tf.float32, [hp.vocab_size, hp.word_embed_size])
            if hp.is_glove:
                self.word_embed_table = embed(len(self.word_w2i), hp.word_embed_size, scope="word_embed")
                self.glove_assign = self.word_embed_table.assign(self.glove_weights)
            else:
                self.word_embed_table = embed(len(self.word_w2i), hp.word_embed_size, scope="word_embed",
                                              trainable=True)
            self.attri_embed_table = embed(2, hp.attri_embed_size, scope="attri_embed", trainable=True)

            # Embedding for encoder input
            self.enc_embed = tf.nn.embedding_lookup(self.word_embed_table, self.x)

            # Embedding for Attributes
            self.attri_embed = tf.nn.embedding_lookup(self.attri_embed_table, self.attributes)
            self.attri_embed = tf.reshape(self.attri_embed, (-1, hp.attri_embed_size))
            # Encoder
            self.enc_outputs = encode(self.enc_embed, self.attri_embed)

            # Embedding for decoder input
            self.decoder_inputs = shift_by_one(self.y)
            self.dec_embed = tf.nn.embedding_lookup(self.word_embed_table, self.decoder_inputs)

            # Decoder
            self.outputs = decode(self.dec_embed, self.enc_outputs, len(self.word_w2i))
            # self.logprobs = tf.log(tf.nn.softmax(self.outputs) + 1e-10)
            self.pred = tf.to_int32(tf.argmax(self.outputs, axis=-1))
            # self.pred_top_5 = tf.nn.top_k(self.outputs, k=5, sorted=True)
            self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)  # Training Scheme
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # it didn't work well with Adadelta
            self.optimizer = tf.train.AdamOptimizer(hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


class DeleteAndRetrieveGraph():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.a = tf.placeholder(tf.int32, shape=(None, hp.ngram))

            # Vocab
            self.word_w2i, _ = load_vocab()

            # Embedding table
            self.glove_weights = tf.placeholder(tf.float32, [hp.vocab_size, hp.word_embed_size])
            if hp.is_glove:
                self.word_embed_table = embed(len(self.word_w2i), hp.word_embed_size, scope="word_embed")
                self.glove_assign = self.word_embed_table.assign(self.glove_weights)
            else:
                self.word_embed_table = embed(len(self.word_w2i), hp.word_embed_size, scope="word_embed",
                                              trainable=True)
            # Embedding for encoder input
            self.enc_embed = tf.nn.embedding_lookup(self.word_embed_table, self.x)

            # Embedding for Attributes
            self.attri_embed = tf.nn.embedding_lookup(self.word_embed_table, self.a)
            # Encoder
            self.enc_outputs = encode(self.enc_embed, self.attri_embed)

            # Embedding for decoder input
            self.decoder_inputs = shift_by_one(self.y)
            self.dec_embed = tf.nn.embedding_lookup(self.word_embed_table, self.decoder_inputs)

            # Decoder
            self.outputs = decode(self.dec_embed, self.enc_outputs, len(self.word_w2i))

            # self.logprobs = tf.log(tf.nn.softmax(self.outputs) + 1e-10)
            self.pred = tf.to_int32(tf.argmax(self.outputs, axis=-1))
            # self.pred_top_5 = tf.nn.top_k(self.outputs, k=5, sorted=True)
            self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.pred, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)  # Training Scheme
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()
