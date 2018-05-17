from hyperparams import Hyperparams as hp
from data import load_sentences, make_dict
import os
from delete import delete

print("Make word dictionary...")
neg_lines = load_sentences(hp.data_path + '/sentiment.train.0', True)
pos_lines = load_sentences(hp.data_path + '/sentiment.train.1', True)
make_dict(neg_lines + pos_lines)

# delete file
print("Make delete file...")
if not os.path.exists(hp.data_path + '/delete'):
    os.makedirs(hp.data_path + '/delete')
    delete(hp.data_path + '/sentiment.train.0', hp.data_path + '/sentiment.train.1', mode='train')
    delete(hp.data_path + '/sentiment.dev.0', hp.data_path + '/sentiment.dev.1', mode='test')
    delete(hp.data_path + '/sentiment.test.0', hp.data_path + '/sentiment.test.1', mode='test')
