from utils import get_ngram, get_tf_idf
from data import load_sentences, make_dict
import os
import codecs
from hyperparams import Hyperparams as hp


def delete(neg_filepath, pos_filepath, mode='train'):
    '''
    divide a sentence into attribute markers and contents.
    '''
    neg_sentences = load_sentences(neg_filepath)
    pos_sentences = load_sentences(pos_filepath)
    if not (os.path.exists(hp.data_path + '/tf_idf.0') and os.path.exists(hp.data_path + '/tf_idf.1')):
        print('making tf_idf dictionary...')
        # n-gram
        neg_ngram_dict = get_ngram(neg_sentences)
        pos_ngram_dict = get_ngram(pos_sentences)
        # relative frequency
        get_tf_idf(neg_ngram_dict, pos_ngram_dict)
    neg_ngrams = dict(x.split('\t') for x in codecs.open(hp.data_path + '/tf_idf.0', 'r').read().splitlines())
    pos_ngrams = dict(x.split('\t') for x in codecs.open(hp.data_path + '/tf_idf.1', 'r').read().splitlines())
    # divide sentence into attribute markers and contents
    # neg:0 pos:1
    print('dividing sentences...')
    for num in ['0', '1']:
        if num == '0':
            sentences = neg_sentences
            attribute_markers = neg_ngrams
        else:
            sentences = pos_sentences
            attribute_markers = pos_ngrams
        with codecs.open(hp.data_path + '/delete/delete.' + mode + '.' + num, 'w', 'utf-8') as fout:
            for sentence in sentences:
                words = sentence.split(' ')
                value_dict = {}
                for i in range(len(words)):
                    for n in range(hp.ngram - 1, 0, -1):
                        if i + n > len(words):
                            continue
                        tmp_attribute_marker = ' '.join(words[i:i + n])
                        if tmp_attribute_marker in attribute_markers:
                            value_dict[tmp_attribute_marker] = attribute_markers[tmp_attribute_marker]
                # if a sentence has attribute_marker
                if len(value_dict) > 0:
                    attribute_marker = max(value_dict, key=value_dict.get)
                    content = sentence.replace(attribute_marker, '')
                    if len(content) > 2:
                        fout.write(sentence + '\t' + content + '\t' + attribute_marker + '\n')


if __name__ == '__main__':
    if not os.path.exists(hp.data_path + '/delete'): os.makedirs(hp.data_path + '/delete')
    delete(hp.data_path + '/sentiment.test.0', hp.data_path + '/sentiment.test.1', mode='test')
