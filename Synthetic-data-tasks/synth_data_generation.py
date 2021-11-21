"""
Provides generating functions for Adding and Temporal Order data
"""
import random
import torch
from tqdm import tqdm

def adding(sequences, n_data):
    full = []
    labels = []
    for _ in tqdm(range(sequences)):
        x = (-1 - 1) * torch.rand(n_data) + 1
        y = torch.zeros(n_data)
        pos_1 = pos_2 = -1
        while pos_1 == pos_2:
            samples = list(random.sample(range(n_data), 2))
            samples.sort()
            pos_1, pos_2 = samples

        y[pos_1] = y[pos_2] = 1
        data = torch.vstack([x, y]).T
        full.append(data)
        label = 0.5 + (x[pos_1] + x[pos_2]) / 4
        labels.append(label)
        
    data = torch.vstack(full).reshape(sequences, n_data, 2)
    labels = torch.tensor(labels)
    return data, labels

def temporal_order(sequences, n_data):
    a = 0
    b = 1
    c = 2
    d = 3
    X = 4
    Y = 5
    n_emb = 1
    
    full = []
    labels = []
    for _ in tqdm(range(sequences)):
        x = [random.choice([a, b, c, d]) for _ in range(n_data)]
        pos_1 = pos_2  = -1
        while pos_1 == pos_2:
            samples = list(random.sample(range(n_data), 2))
            samples.sort()
            pos_1, pos_2 = samples

        val_1 = random.choice([X, Y])
        val_2 = random.choice([X, Y])
        x[pos_1] = val_1
        x[pos_2] = val_2

        if (val_1 == X) and (val_2 == X):
            label = 0
        elif (val_1 == X) and (val_2 == Y):
            label = 1
        elif (val_1 == Y) and (val_2 == X):
            label = 2
        else:
            label = 3
            
        x = torch.tensor(x)
            
        full.append(x)
        labels.append(label)
        
    data = torch.vstack(full).reshape(sequences, n_data, n_emb)
    labels = torch.tensor(labels)
    return data, labels


def main():

    n_sequences = {
        'train': 100000,
        'val': 1000,
        'test': 1000
    }
    seq_lenths = [2**8]#, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]
    for seq_len in seq_lenths:
        print(f"Generation sequences for length={seq_len}")
        for ds, n_seq in n_sequences.items():
            data, labels = adding(n_seq, seq_len)
            torch.save(data, f'adding_{seq_len}_{ds}.pt')
            torch.save(labels, f'adding_{seq_len}_{ds}_target.pt')

        for ds, n_seq in n_sequences.items():
            data, labels = temporal_order(n_seq, seq_len)
            torch.save(data, f'order_{seq_len}_{ds}.pt')
            torch.save(labels, f'order_{seq_len}_{ds}_target.pt')

if __name__ == "__main__":
    main()
