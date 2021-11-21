import torch
from tqdm import tqdm
import numpy as np
import csv
from torch.nn.utils.rnn import pad_sequence
import sys

# Building the vocab
with open("./data/listops-1000/basic_test.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    next(tsvreader, None)  # skip the headers
    for line in tsvreader:
        line = line[0]
        line = line.replace(']', 'X').replace('(', '').replace(')', '')
        seq = line.split(' ')
        seq = list(filter(None, seq))
        all = list(set(seq))
        all.sort()
        print(all)
        # all = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[MAX', '[MED', '[MIN', '[SM', 'X']
        break

vocab_size = len(all) + 1 # 16

ch2idx = {
    x: i
    for i, x in enumerate(all)
}
ch2idx['<PAD>'] = vocab_size - 1
max_seq = 1999
print(ch2idx)

for part in ['test', 'val', 'train']:
    print(f'Starting {part}')
    sources = []
    targets = []
    sources.append(torch.Tensor([0 for i in range(max_seq)]))
    with open(f'./data/listops-1000/basic_{part}.tsv') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # Skip the headers
        total = 96000 if part == 'train' else 2000
        for line in tqdm(tsvreader, total=total):
            targ = line[1]
            line = line[0]
            line = line.replace(']', 'X').replace('(', '').replace(')', '')
            seq = line.split(' ')
            seq = list(filter(None, seq))
            mapped_seq = [ch2idx[token] for token in seq]
            # context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            sources.append(torch.Tensor(mapped_seq))
            targets.append(int(targ))

    final_tensor = pad_sequence(sources, padding_value=ch2idx['<PAD>']).T[1:]
    final_targets = torch.tensor(targets, dtype=torch.int32)
    print('data shape', final_tensor.shape)
    print('targets shape', final_targets.shape)
    # print(final_tensor[0][-10:])
    torch.save(final_targets, f'target_{part}_clean.pt')
    torch.save(final_tensor, f'{part}_clean.pt')

    r_t = torch.load(f'target_{part}_clean.pt')
    r_d = torch.load(f'{part}_clean.pt')
    print('data shape', r_d.shape)
    print('targets shape', r_t.shape)