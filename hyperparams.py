class Hyperparams:
    # delete
    ngram_size = 5000
    tf_idf_threshold = 15
    ngram = 5

    # retrieve
    min_distance = 3
    max_candidates = 1000

    # neural
    word_embed_size = 200
    attri_embed_size = 512
    gru_size = 512
    batch_size = 256
    vocab_size = 8000
    max_len = 10
    num_epochs = 50
    lr = 0.0001
    neural_mode = 'delete_only'
    is_earlystopping = True
    is_glove = True

    # path
    data_path = 'datas/yelp'
