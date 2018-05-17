from generate_baseline import retrieve_only, template_based
from hyperparams import Hyperparams as hp
from delete import delete
from inference import inference
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', action='store', dest='mode', type=str, default='retrieve_only',
                        help='Enter generate mode')
    par_args = parser.parse_args()
    mode = par_args.mode

    # generate mode
    if not os.path.exists(hp.data_path + '/generate'):
        os.makedirs(hp.data_path + '/generate')

    if mode == 'retrieve_only':
        retrieve_only()
    elif mode == 'template_based':
        template_based()
    # before using neural model, you have to train first
    elif mode == 'delete_only':
        hp.neural_mode = mode
        inference(mode)
    elif mode == 'delete_and_retrieve':
        hp.neural_mode = mode
        inference(mode)
    else:
        print('Enter Correct Mode Name...')
