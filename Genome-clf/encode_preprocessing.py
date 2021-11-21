import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Bio import SeqIO

bases = {"a": 0, "g":1, "c":2, "t":3, "<PAD>":4}

def merge():
    input_file_human = "./data/encode/NONCODEv6_human.fa"
    fasta_sequences = SeqIO.parse(open(input_file_human), 'fasta')
    human_sequences = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        human_sequences.append(sequence)

    human_df = pd.DataFrame(human_sequences, columns=["sequence"])
    human_df["length"] = human_df['sequence'].apply(lambda x: len(x))
    human_df["sequence"] = human_df['sequence'].apply(lambda x: x.lower())
    human_df = human_df[human_df["length"] > 5000]
    human_df["class"] = 1

    input_file_fruitfly = "./data/encode/NONCODEv6_fruitfly.fa"
    fasta_sequences_ff = SeqIO.parse(open(input_file_fruitfly), 'fasta')
    fruitfly_sequences = []
    for fasta in fasta_sequences_ff:
        name, sequence = fasta.id, str(fasta.seq)
        fruitfly_sequences.append(sequence)

    fruitfly_df = pd.DataFrame(fruitfly_sequences, columns=["sequence"])
    fruitfly_df["length"] = fruitfly_df['sequence'].apply(lambda x: len(x))
    fruitfly_df["sequence"] = fruitfly_df['sequence'].apply(lambda x: x.lower())
    fruitfly_df = fruitfly_df[fruitfly_df["length"] > 5000]
    fruitfly_df["class"] = 0

    df = pd.concat((human_df, fruitfly_df), axis=0)
    return df

def dna_to_vector(data, bases):
    """
    transform DNA to vector
    :param texts:
    :param token_index: vocabulary: {8, 'Y': 9, 'z': 10, ' ': 11}
    :return: encoded sequence [8, 12, 34, 56, 14]
    """
    encoded_dna = []
    sequences = data.sequence.values
    lengths = data.length.values
    for i in range(len(sequences)):
        encodes = [lengths[i]]
        encodes = np.array(encodes+[bases[ch] for ch in sequences[i]])
        encoded_dna.append(encodes)
    return encoded_dna


def get_dna_data(max_len):
    # get DNA sequences that longer than 20000

    data_df = merge()

    # conver DNA sequences to vectors
    X = dna_to_vector(data_df, bases)
    y = data_df["class"].values

    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, value=bases['<PAD>'],
                                                            padding="post", truncating='post')

    # split train test and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    test_lengths = torch.tensor(X_test[:, 0])
    val_lengths = torch.tensor(X_val[:, 0])

    X_train = torch.tensor(X_train[:, 1:])
    X_test = torch.tensor(X_test[:, 1:])
    X_val = torch.tensor(X_val[:, 1:])
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    y_val = torch.tensor(y_val)

    print(test_lengths)
    print(len(y_train[y_train==0]), len(y_test[y_test==0]))

    print(X_train.shape, X_test.shape, X_val.shape)

    torch.save(X_train, './data/encode/Encode16384_train.pt')
    torch.save(y_train, './data/encode/Encode16384_train_targets.pt')

    torch.save(X_test, './data/encode/Encode16384_test.pt')
    torch.save(y_test, './data/encode/Encode16384_test_targets.pt')

    torch.save(X_val, './data/encode/Encode16384_val.pt')
    torch.save(y_val, './data/encode/Encode16384_val_targets.pt')

    torch.save(test_lengths, './data/encode/Encode16384_test_lengths.pt')
    return X_train, y_train, X_test, y_test, X_val, y_val

get_dna_data(16385)