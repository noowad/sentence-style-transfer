import os
import codecs
from hyperparams import Hyperparams as hp
from delete import delete
from data import load_sentences
import random
import numpy as np
from utils import levenshtein_distance
import tensorflow as tf
import tensorflow_hub as hub


# generate by retrieve only method
def retrieve_only(dist_mode='levenshtein'):

    print('retrieve_only with ' + dist_mode + ' distance...')
    for num in ['0', '1']:
        if num == '0':
            neg_lines = load_sentences(hp.data_path + '/delete/delete.test.0', False)
            pos_lines = load_sentences(hp.data_path + '/delete/delete.train.1', False)
            neg_sentences_dict = dict(x.split('\t')[:2] for x in neg_lines)
            pos_sentences_dict = dict(x.split('\t')[:2] for x in pos_lines)
            sentences1 = neg_sentences_dict
            sentences2 = pos_sentences_dict
        else:
            neg_lines = load_sentences(hp.data_path + '/delete/delete.train.0', False)
            pos_lines = load_sentences(hp.data_path + '/delete/delete.test.1', False)
            neg_sentences_dict = dict(x.split('\t')[:2] for x in neg_lines)
            pos_sentences_dict = dict(x.split('\t')[:2] for x in pos_lines)
            sentences1 = pos_sentences_dict
            sentences2 = neg_sentences_dict
        with codecs.open(hp.data_path + '/generate/retrieve_only.test.' + num, 'w', 'utf-8') as fout:
            # Levenshtein distance
            if dist_mode == 'levenshtein':
                for sentence1 in sentences1:
                    dist_dict = {}
                    # Search up to hp.max_candidates randomly.
                    frag_sentences2 = random.sample(sentences2.keys(), hp.max_candidates)
                    for sentence2 in frag_sentences2:
                        # distance between pos_content and neg_content
                        dist_dict[sentence2] = levenshtein_distance(sentences1[sentence1], sentences2[sentence2])
                    nearest_sentence = min(dist_dict, key=dist_dict.get)
                    fout.write("- expected: " + sentence1 + "\n")
                    fout.write("- got: " + nearest_sentence + "\n\n")
                    fout.flush()

            # Embedding distance between sentence1,sentence2 by using "universal sentence encoder[1]"
            # but it's too slow and not good performance
            if dist_mode == 'embedding':
                embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
                with tf.Session() as session:
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                    embedded_sentences1 = session.run(embed(sentences1.values()))
                    for sentence1, embedded_sentence1 in zip(sentences1.keys(), embedded_sentences1):
                        dist_dict = {}
                        # Search up to hp.max_candidates randomly.
                        frag_sentences2 = random.sample(sentences2.keys(), hp.max_candidates)
                        frag_contents2 = []
                        for frag_sentence2 in frag_sentences2:
                            frag_contents2.append(sentences2[frag_sentence2])
                        embedded_sentences2 = session.run(embed(frag_contents2))
                        for idx, embedded_sentence2 in enumerate(embedded_sentences2):
                            dist_dict[idx] = np.inner(embedded_sentence1, embedded_sentence2)
                        nearest_idx = max(dist_dict, key=dist_dict.get)
                        nearest_sentence = frag_sentences2[nearest_idx]
                        fout.write("- expected: " + sentence1 + "\n")
                        fout.write("- got: " + nearest_sentence + "\n\n")
                        fout.flush()

# generate by template based method
def template_based():
    if not os.path.exists(hp.data_path + '/generate'):
        os.makedirs(hp.data_path + '/generate')
    print('template_based...')
    for num in ['0', '1']:
        if num == '0':
            neg_lines = load_sentences(hp.data_path + '/delete/delete.test.0', False)
            pos_lines = load_sentences(hp.data_path + '/delete/delete.train.1', False)
            neg_sentences_contents_dict = dict(x.split('\t')[:2] for x in neg_lines)
            pos_sentences_contents_dict = dict(x.split('\t')[:2] for x in pos_lines)
            pos_sentences_marker_dict = dict([x.split('\t')[0], x.split('\t')[2]] for x in pos_lines)
            sentences1 = neg_sentences_contents_dict
            sentences2 = pos_sentences_contents_dict
            marker2 = pos_sentences_marker_dict
        else:
            neg_lines = load_sentences(hp.data_path + '/delete/delete.train.0', False)
            pos_lines = load_sentences(hp.data_path + '/delete/delete.test.1', False)
            neg_sentences_contents_dict = dict(x.split('\t')[:2] for x in neg_lines)
            pos_sentences_contents_dict = dict(x.split('\t')[:2] for x in pos_lines)
            neg_sentences_marker_dict = dict([x.split('\t')[0], x.split('\t')[2]] for x in neg_lines)
            sentences1 = pos_sentences_contents_dict
            sentences2 = neg_sentences_contents_dict
            marker2 = neg_sentences_marker_dict
        with codecs.open(hp.data_path + '/generate/template_based.test.' + num, 'w', 'utf-8') as fout:
            for sentence1 in sentences1:
                dist_dict = {}
                # Search up to hp.max_candidates randomly.
                frag_sentences2 = random.sample(sentences2.keys(), hp.max_candidates)
                sentence1_content = sentences1[sentence1]
                for sentence2 in frag_sentences2:
                    # distance between pos_content and neg_content
                    dist_dict[sentence2] = levenshtein_distance(sentence1_content, sentences2[sentence2])
                min_sentence = min(dist_dict, key=dist_dict.get)
                nearest_marker = marker2[min_sentence]
                sentence1_list = sentence1.split(' ')
                sentence1_content_list = sentences1[sentence1].split(' ')
                # Insert attribute markers in contents
                index = 0
                for idx in range(len(sentence1_list)):
                    if sentence1_list[idx] != sentence1_content_list[idx]:
                        index = idx
                        break
                generated_sentence = ' '.join(sentence1_content_list[:index]) + ' ' + \
                                     nearest_marker + ' ' + ' '.join(sentence1_content_list[index:])
                generated_sentence = generated_sentence.replace('  ', ' ')
                fout.write("- expected: " + sentence1 + "\n")
                fout.write("- got: " + generated_sentence + "\n\n")
                fout.flush()
