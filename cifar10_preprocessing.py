import warnings

warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import sys
import tensorflow as tf
from PIL import Image
import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor()
     ]
)

batch_size = 200

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# Building the vocab
for image, target in trainloader:
    unique = image.unique(sorted=True)
    assert len(unique) == 256, 'Vocab is not full'
    break

vocab_size = len(unique)  # 256

pix2idx = {
    float(x): i
    for i, x in enumerate(unique)
}
# print(pix2idx)

all_imgs = []
all_targets = []
for image, target in tqdm(trainloader):
    image = image.squeeze(1).flatten(start_dim=1)
    # print(image.size())
    for im_idx in range(image.size(0)):
        img = [pix2idx[float(token)] for token in image[im_idx]]
        all_imgs.append(torch.Tensor(img))
        all_targets.append(target[im_idx])

all_imgs = torch.stack(all_imgs)
all_targets = torch.stack(all_targets)

print('ALL IMGS TENSOR SIZE:', all_imgs.size())
print('ALL TRGS TENSOR SIZE:', all_targets.size())

torch.save(all_imgs, 'cifar10_train_vocab.pt')
torch.save(all_targets, 'cifar10_train_targets_vocab.pt')

all_imgs = []
all_targets = []
for image, target in tqdm(testloader):
    image = image.squeeze(1).flatten(start_dim=1)

    for im_idx in range(image.size(0)):
        img = [pix2idx[float(token)] for token in image[im_idx]]
        all_imgs.append(torch.Tensor(img))
        all_targets.append(target[im_idx])

all_imgs = torch.stack(all_imgs)
all_targets = torch.stack(all_targets)

print('ALL IMGS TENSOR SIZE:', all_imgs.size())
print('ALL TRGS TENSOR SIZE:', all_targets.size())

torch.save(all_imgs, 'cifar10_test_vocab.pt')
torch.save(all_targets, 'cifar10_test_targets_vocab.pt')