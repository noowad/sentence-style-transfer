# A Tensorflow Implementation of "Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer"
## Requirements
- Python 2.7
- Tensorflow >=1.3
- Numpy
- tqdm
## Notes
- This is an implementation of ["Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer"](https://arxiv.org/abs/1804.06437 "").
- This repository contains the results of Yelp dataset.
- I implemented this for Japanese-sentence style transfer, but I could not reveal Japanese datasets because of copyright.
## Differences with the original paper
- I used pre-trained word embedding of 200-dimensions [GloVe](https://nlp.stanford.edu/projects/glove/ "glove").
- I used Adam optimizer for neural model's training. In my case, loss did not converge with Adadelta optimizer.
- I modified the encoder of DeleteOnly by adding embedding of contents and embedding of attribute (not concatenation).
- I didn't use Maxout activation function.
- I used greedy search instead of beam search.
- I tested on Yelp dataset only.
## Execution
- STEP 0. Download 200-D GloVe vectors (from https://nlp.stanford.edu/projects/glove/) and copy it to datas/yelp/glove.
- STEP 1. Adjust hyper parameters in `hyperparams.py`. Of course, you do not need to modify it!
- STEP 2. Run `python preprocess.py` for making word dictionary and delete file.
- STEP 3. Run `python train.py -mode delete_only` and `python train.py -mode delete_and_retrieve`  for training of neural models (DeleteOnly and DeleteAndRetrieve).
- STEP 4. Run `python main.py -mode 'generate mode'` for generating sentences.  
'generate mode' includes 4 modes; 'retrieve_only', 'template_based', 'delete_only' and 'delete_and_retrieve'. 
## Result Examples
Generated examples are contained in datas/yelp/generate. Check it!
## References
- The original codes: ["lijuncen/Sentiment-and-Style-Transfer"](https://github.com/lijuncen/Sentiment-and-Style-Transfer "")
- I referenced codes of [Kyubyong](https://github.com/Kyubyong "") for Tensorflow coding styles.
