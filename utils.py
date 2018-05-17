# coding:utf-8
from hyperparams import Hyperparams as hp
import random
import codecs
import numpy as np
import tensorflow as tf


def get_ngram(sentences):
    ngram_dict = {}
    for sentence in sentences:
        words = sentence.split(' ')
        for n in range(1, hp.ngram):
            for l in range(0, len(words) - n + 1):
                ngrams = ' '.join(words[l:l + n])
                if ngrams not in ngram_dict:
                    ngram_dict[ngrams] = 1
                else:
                    ngram_dict[ngrams] += 1
    return ngram_dict


# n-gram relative frequency
def get_tf_idf(neg_dict, pos_dict):
    '''
    This is not a definition of TF-IDF but a definition of n-gram relative frequency from the paper.
    but I named it TF-IDF because it has a similar meaning.
    '''
    for num in ['0', '1']:
        tf_idf = {}
        for i in neg_dict.keys():
            if i not in pos_dict:
                tmp_tf_idf = (neg_dict[i] + 1.0)
            else:
                tmp_tf_idf = (neg_dict[i] + 1.0) / (pos_dict[i] + 1.0)
            if tmp_tf_idf > hp.tf_idf_threshold:
                tf_idf[i] = tmp_tf_idf
        sorted_tf_idf = sorted(tf_idf.iteritems(), key=lambda d: d[1], reverse=True)[:hp.ngram_size]

        with codecs.open(hp.data_path + '/tf_idf.' + num, 'w', 'utf-8') as fout:
            for i in sorted_tf_idf:
                fout.write(str(i[0]) + '\t' + str(i[1]) + '\n')

        neg_dict, pos_dict = pos_dict, neg_dict


def levenshtein_distance(sentence1, sentence2):
    '''
    levenshtein distance between sentence1, sentence2
    :param sentence1: sentence string
    :param sentence2: sentence string
    '''
    sentence1, sentence2 = sentence1.split(' '), sentence2.split(' ')

    if len(sentence1) > len(sentence2):
        sentence1, sentence2 = sentence2, sentence1
    # dynamic programming
    distances = range(len(sentence1) + 1)
    for num2, word2 in enumerate(sentence2):
        distances_ = [num2 + 1]
        for num1, word1 in enumerate(sentence1):
            if word1 == word2:
                distances_.append(distances[num1])
            else:
                distances_.append(1 + min((distances[num1], distances[num1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one
      so that it becomes the decoder inputs.

    Args:
      inputs: A 2d tensor with shape of [N, T]

    Returns:
      A 2d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1]), inputs[:, :-1]), 1)


def noising_attribute_marker(attribute_markers):
    # remove 1 word from sentence
    def modify_sentence(sentence):
        li = sentence.split()
        li.remove(np.random.choice(li))
        return ' '.join(li)

    noised_attribute_markers = []
    word_dict = {}
    # generate word dictionary
    for attribute_marker in attribute_markers:
        for i in attribute_marker.split():
            if i not in word_dict:
                word_dict[i] = 0
    # noising
    for attribute_marker in attribute_markers:
        noised_list = []
        attribute_marker_list = attribute_marker.split()
        if len(attribute_marker_list) != 1:
            for i in attribute_marker_list:
                if np.random.choice([0, 1], p=[0.9, 0.1]) == 1:
                    i = np.random.choice(word_dict.keys())
                noised_list.append(i)
            noised_attribute_marker = ' '.join(noised_list)
            if noised_attribute_marker == attribute_marker:
                noised_attribute_marker = modify_sentence(attribute_marker)
        else:
            noised_attribute_marker = np.random.choice(word_dict.keys())
        noised_attribute_markers.append(noised_attribute_marker)
    return noised_attribute_markers

