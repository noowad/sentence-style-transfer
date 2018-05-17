import codecs
import numpy as np
from utils import noising_attribute_marker
from hyperparams import Hyperparams as hp
from utils import levenshtein_distance


def make_dict(sentences, vocab_filepath='datas/yelp/yelp.dict'):
    glove_lines = open(hp.data_path + '/glove/glove.6B.' + str(hp.word_embed_size) + 'd.txt', 'r').read().splitlines()
    glove_words = dict([line.split(' ')[0], i] for i, line in enumerate(glove_lines))

    word_dict = {}
    for sentence in sentences:
        for word in sentence.split(' '):
            if word in glove_words:
                if word_dict.get(word) is None:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
    # write vocab_file
    with codecs.open(vocab_filepath, 'w', 'utf-8') as fout:
        fout.write('<PAD>\t10000000\n')
        fout.write('<UNK>\t10000000\n')
        fout.write('<START>\t10000000\n')
        fout.write('<END>\t10000000\n')
        for num, word in enumerate(sorted(word_dict.iteritems(), key=lambda d: d[1], reverse=True)):
            if word[1] > 0:
                fout.write(str(word[0]) + '\t' + str(word[1]) + '\n')


def load_sentences(filepath='', is_preprocess=True):
    sentences = codecs.open(filepath, 'r', 'utf-8').read().splitlines()
    # Remove one-word sentences. (includes period)
    if is_preprocess:
        sentences = list(filter(lambda sen: len(sen.split(' ')) > 2, sentences))
    return sentences


def load_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.data_path + '/yelp.dict', 'r', 'utf-8').read().splitlines()][
            :hp.vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


# attribute
# 0: negative 1:positive
# for DeleteOnly
def create_data(lines, attribute=0, mode='train'):
    word2idx, idx2word = load_vocab()
    sentences = [x.split('\t')[0] for x in lines]
    contents = [x.split('\t')[1] for x in lines]
    if mode == 'train':
        attribute_markers = [x.split('\t')[2] for x in lines]
    else:
        attribute_markers = retrieve_nearest_marker(lines, attribute)
    # attribute_marker denoising
    if hp.neural_mode == 'delete_and_retrieve':
        attribute_markers = noising_attribute_marker(attribute_markers)

    # Index
    x_list, y_list, sources, targets, idx_attributes, attributes = [], [], [], [], [], []
    for source_sent, target_sent, attribute_marker in zip(contents, sentences, attribute_markers):
        x = [word2idx.get(word, 1) for word in (source_sent + ' <END>').split()]
        y = [word2idx.get(word, 1) for word in (target_sent + ' <END>').split()]
        a = np.array([word2idx.get(word, 1) for word in (attribute_marker + ' <END>').split()])
        if max(len(x), len(y)) <= hp.max_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source_sent)
            targets.append(target_sent)
            # for attribute marker
            if len(a) <= hp.ngram:
                idx_attributes.append(np.array(a))
                attributes.append(attribute_marker)
    # Pad
    X = np.zeros([len(x_list), hp.max_len], np.int32)
    Y = np.zeros([len(y_list), hp.max_len], np.int32)
    A = np.zeros([len(attributes), hp.ngram], np.int32)
    for i, (x, y, a) in enumerate(zip(x_list, y_list, idx_attributes)):
        X[i] = np.lib.pad(x, [0, hp.max_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.max_len - len(y)], 'constant', constant_values=(0, 0))
        A[i] = np.lib.pad(a, [0, hp.ngram - len(a)], 'constant', constant_values=(0, 0))
    attribute_labels = [attribute] * len(attributes)
    return X, Y, A, attribute_labels, sources, targets, attributes


# pre-trained GloVe
def load_glove():
    lines = open(hp.data_path + '/glove/glove.6B.' + str(hp.word_embed_size) + 'd.txt', 'r').read().splitlines()
    vocabs = [line.split('\t')[0] for line in open(hp.data_path + '/yelp.dict', 'r').read().splitlines()][
             :hp.vocab_size]
    words = {}
    weights = []
    for idx, line in enumerate(lines):
        parts = line.split(' ')
        words[parts[0]] = idx
        weights.append(np.asarray(parts[1:], dtype=np.float32))
    aligned_weights = []
    for idx, vocab in enumerate(vocabs):
        if vocab in words:
            aligned_weights.append(weights[words[vocab]])
        else:
            if idx == 0:
                aligned_weights.append(np.zeros(hp.word_embed_size))
            else:
                aligned_weights.append(np.random.randn(hp.word_embed_size))
    return np.asarray(aligned_weights, dtype=np.float32)


def retrieve_nearest_marker(lines1, attribute):
    if attribute == 0:
        lines2 = load_sentences(hp.data_path + '/delete/delete.test.0', False)
    else:
        lines2 = load_sentences(hp.data_path + '/delete/delete.test.1', False)
    sentences_contents_dict1 = dict(x.split('\t')[:2] for x in lines1)
    sentences_contents_dict2 = dict(x.split('\t')[:2] for x in lines2)
    sentences_marker_dict = dict([x.split('\t')[0], x.split('\t')[2]] for x in lines2)
    sentences1 = sentences_contents_dict1
    sentences2 = sentences_contents_dict2
    marker = sentences_marker_dict
    attribute_markers = []
    for sentence1 in sentences1:
        dist_dict = {}
        sentence1_content = sentences1[sentence1]
        for sentence2 in sentences2.keys():
            # distance between pos_content and neg_content
            dist_dict[sentence2] = levenshtein_distance(sentence1_content, sentences2[sentence2])
        min_sentence = min(dist_dict, key=dist_dict.get)
        # nearest_marker
        attribute_markers.append(marker[min_sentence])
    return attribute_markers
