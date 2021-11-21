import pandas as pd
import numpy as np
import torch
import tensorflow_datasets as tfds
import tensorflow as tf


def get_char_index(train_df, test_df=None, val_df=None):
    """
    get character vocabulary
    :param train_df: training texts
    :param test_df: test texts
    :param val_df: validation texts
    :return:
    """
    df = pd.concat([train_df, val_df, test_df])
    texts = df.x.values
    chars = [char for text in texts for char in text]
    chars = tuple(set(chars))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    vocab_size = len(chars)+1
    char2int['<PAD>'] = vocab_size - 1
    return char2int, vocab_size


def text_to_sequence(texts, token_index):
    """
    transform text to vector
    :param texts:
    :param token_index: vocabulary: {8, 'Y': 9, 'z': 10, ' ': 11}
    :return: encoded sequence [8, 12, 34, 56, 14]
    """
    encoded_sequences = []
    for text in texts:
        encodes = np.array([token_index[ch] for ch in text])
        encoded_sequences.append(encodes)
    return encoded_sequences


def produce_imdb(max_len):
    """Get dataset from  imdb tfds. converts into src/tgt tensor pairs."""
    data = tfds.load('imdb_reviews')
    train_df = tfds.as_dataframe(data['train'])
    valid_df = tfds.as_dataframe(data['test'])
    test_df = tfds.as_dataframe(data['test'])
    train_df = train_df
    test_df = test_df
    valid_df = valid_df
    train_df.columns = ["y", "x"]
    test_df.columns = ["y", "x"]
    valid_df.columns = ["y", "x"]
    train_df.x = train_df.x.apply(lambda x: str(x))
    test_df.x = test_df.x.apply(lambda x: str(x))
    valid_df.x = valid_df.x.apply(lambda x: str(x))

    token_index, char_vocab = get_char_index(train_df, test_df)
    print(token_index)
    X_train = text_to_sequence(train_df.x.values, token_index)
    X_test = text_to_sequence(test_df.x.values, token_index)
    #print(X_train[2])
    sequences = [len(s) for s in X_train]
    print(f"average sequence length {np.mean(sequences)}")
    # zero padding
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len, value=token_index['<PAD>'],
                                                            padding="post", truncating="post")
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len, value=token_index['<PAD>'],
                                                           padding="post", truncating="post")
    y_train = train_df["y"]
    y_test = test_df["y"]

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    X_val = X_test
    y_val = y_test
    print(X_test)

    print(X_train.shape, X_test.shape)

    torch.save(X_train, 'data/IMDB_train.pt')
    torch.save(y_train, 'data/IMDB_train_targets.pt')

    torch.save(X_test, 'data/IMDB_test.pt')
    torch.save(y_test, 'data/IMDB_test_targets.pt')
    return X_train, y_train, X_test, y_test, X_val, y_val


produce_imdb(1024)