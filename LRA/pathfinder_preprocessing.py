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
from torchvision import datasets, transforms

ORIGINAL_DATA_DIR_32 = 'data/pathfinder32/'
easy = 'curv_baseline'
medium = 'curv_contour_length_9'
hard = 'curv_contour_length_14'

metafiles_easy = os.listdir(os.path.join(ORIGINAL_DATA_DIR_32, easy, 'metadata'))
metafiles_medium = os.listdir(os.path.join(ORIGINAL_DATA_DIR_32, medium, 'metadata'))
metafiles_hard = os.listdir(os.path.join(ORIGINAL_DATA_DIR_32, hard, 'metadata'))
print('Number of metafiles: easy - {}, medium - {}, hard - {}'.format(
    len(metafiles_easy),
    len(metafiles_medium),
    len(metafiles_hard)
))

trans = transforms.Compose([transforms.ToTensor()])


# Building the vocabulary
def get_unique_pixels(file_path, complexity=easy):
    """Read the input data out of the source files."""
    meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split('\n')[:-1]
    pix = []
    for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_32, complexity, m_example[0], m_example[1])

        # sys.exit()
        try:
            img = Image.open(image_path)
            pix.append(trans(img).unique())
        except:
            print('Problem in {}/{}'.format(m_example[0], m_example[1]))

    pix = torch.cat(pix, 0).unique(sorted=True)
    return pix


all_pixels = []
for complex in [easy, medium, hard]:
    path_to_metafiles = os.path.join(ORIGINAL_DATA_DIR_32, complex, 'metadata')
    metafiles = os.listdir(path_to_metafiles)
    metafiles.sort()
    print(len(metafiles))
    for metafile in tqdm(metafiles):
        pixels = get_unique_pixels(path_to_metafiles + '/{}'.format(metafile), complexity=complex)
        all_pixels.append(pixels)

all_pixels = torch.cat(all_pixels, 0).unique(sorted=True)

pix2idx = {
    float(x): i
    for i, x in enumerate(all_pixels)
}
print(len(pix2idx))


def get_image_target(file_path, complexity=easy):
    """Read the input data out of the source files."""
    meta_examples = tf.io.read_file(file_path).numpy().decode('utf-8').split('\n')[:-1]
    imgs = []
    targets = []
    # print('Number of images:', len(meta_examples))
    for m_example in meta_examples:
        m_example = m_example.split(' ')
        image_path = os.path.join(ORIGINAL_DATA_DIR_32, complexity, m_example[0], m_example[1])
        try:
            img = Image.open(image_path)
            img = trans(img).squeeze(0).flatten()
            X = torch.tensor([pix2idx[float(token)] for token in img]).to(torch.int32)
            imgs.append(X)
            targets.append(int(m_example[3]))
        except:
            print('Problem in {}/{}'.format(m_example[0], m_example[1]))

    imgs = torch.stack(imgs)
    targets = torch.tensor(targets, dtype=torch.int32)
    return imgs, targets


all_imgs = []
all_targets = []
for complex in [easy, medium, hard]:
    path_to_metafiles = os.path.join(ORIGINAL_DATA_DIR_32, complex, 'metadata')
    metafiles = os.listdir(path_to_metafiles)
    metafiles.sort()
    # sys.exit()
    for metafile in tqdm(metafiles[20:]):
        imgs, targets = get_image_target(path_to_metafiles + '/{}'.format(metafile), complexity=complex)
        all_imgs.append(imgs)
        all_targets.append(targets)

all_imgs = torch.cat(all_imgs, 0)
all_targets = torch.cat(all_targets, 0)

print('ALL IMGS TENSOR SIZE:', all_imgs.size())
print('ALL TRGS TENSOR SIZE:', all_targets.size())

torch.save(all_imgs, 'pathfinder32_all_train.pt')
torch.save(all_targets, 'pathfinder32_all_train_targets.pt')

print('_' * 60)

all_imgs = []
all_targets = []
for complex in [easy, medium, hard]:
    path_to_metafiles = os.path.join(ORIGINAL_DATA_DIR_32, complex, 'metadata')
    metafiles = os.listdir(path_to_metafiles)
    metafiles.sort()
    for metafile in tqdm(metafiles[:10]):
        imgs, targets = get_image_target(path_to_metafiles + '/{}'.format(metafile), complexity=complex)
        all_imgs.append(imgs)
        all_targets.append(targets)

all_imgs = torch.cat(all_imgs, 0)
all_targets = torch.cat(all_targets, 0)

print('ALL IMGS TENSOR SIZE:', all_imgs.size())
print('ALL TRGS TENSOR SIZE:', all_targets.size())

torch.save(all_imgs, 'pathfinder32_all_test.pt')
torch.save(all_targets, 'pathfinder32_all_test_targets.pt')

all_imgs = []
all_targets = []
for complex in [easy, medium, hard]:
    path_to_metafiles = os.path.join(ORIGINAL_DATA_DIR_32, complex, 'metadata')
    metafiles = os.listdir(path_to_metafiles)
    metafiles.sort()
    for metafile in tqdm(metafiles[10:20]):
        imgs, targets = get_image_target(path_to_metafiles + '/{}'.format(metafile), complexity=complex)
        all_imgs.append(imgs)
        all_targets.append(targets)

all_imgs = torch.cat(all_imgs, 0)
all_targets = torch.cat(all_targets, 0)

print('ALL IMGS TENSOR SIZE:', all_imgs.size())
print('ALL TRGS TENSOR SIZE:', all_targets.size())

torch.save(all_imgs, 'pathfinder32_all_val.pt')
torch.save(all_targets, 'pathfinder32_all_val_targets.pt')

# For Inference part
# taking the names of the testset images
import pandas as pd

df = pd.DataFrame()

all_imgs_paths = []
for complex in [easy, medium, hard]:
    path_to_metafiles = os.path.join(ORIGINAL_DATA_DIR_32, complex, 'metadata')
    metafiles = os.listdir(path_to_metafiles)
    metafiles.sort()
    for metafile in tqdm(metafiles[:10]):

        meta_examples = tf.io.read_file(path_to_metafiles + '/{}'.format(metafile)).numpy().decode('utf-8').split('\n')[
                        :-1]
        # print('Number of images:', len(meta_examples))
        for m_example in meta_examples:
            m_example = m_example.split(' ')
            image_path = os.path.join(ORIGINAL_DATA_DIR_32, complex, m_example[0], m_example[1])
            all_imgs_paths.append(image_path)

df['img_path'] = all_imgs_paths

df.to_csv('img_paths.csv', index=False)
