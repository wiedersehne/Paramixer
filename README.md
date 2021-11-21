# Paramixer

## Install requirements.txt
All the system requirements (used packages and their versions) are provided in requirements.txt.

The most important packages are torch-sparse and torch-geometric. The batched sparse multiplication described in the main paper is built upon torch_sparse.spmm and torch_geometric.DataLoader functions. Make sure they are installed. 

## Synthetic task
To reproduce the synthetic data experiment results, you have to generate the sequences data via ***synth_data_generation.py***. Based on the set sequence length, it will create tensors for both the Adding and Temporal Order problems.  By default it iterates over all sequences lengths: `[2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15]`. The script generates six tensors for each length and problem and stores them in the default folder in the following format:

`{problem}_{n_vec}_train.pt`
`{problem}_{n_vec}_train_targets.pt`
`{problem}_{n_vec}_test.pt`
`{problem}_{n_vec}_test_targets.pt`
`{problem}_{n_vec}_val.pt`
`{problem}_{n_vec}_val_targets.pt`

To train PSF-Attn and X-formers, you can just run ***synth_training.py***. You can transfer to a different task by changing the following settings:

    cfg_model = config['order']['models']['Transformer']  
    cfg_training = config['order']['training']

We provide used configurations for each model on each task in ***synthetic_training_config.py***. For instance, we use the following setting for PSF-Attn on Adding problem.

    "Paramixer":{
                "add_init_linear_layer": True,
                "vocab_size": 1,
                "embedding_size": 32,
                "dim": 32,
                "pooling_type": "FLATTEN",
                "head": ['linear'],
                "n_class": 1,
                "n_channels_V": 32,
                "use_cuda": True,
                "use_residuals": True,
                "use_pos_embedding": False,
                "problem": "adding",
                "protocol": "chord",
                'n_layers': 2
            },

## Long Range Arena
The data set for the LRA benchmark is publicly available. The information about data and the download link can be found in the official GitHub repository: https://github.com/google-research/long-range-arena.

### Data preparation

To download the LRA dataset (~7.7GB) you have to run the following command:
`wget https://storage.googleapis.com/long-range-arena/lra_release.gz`
As a result, it will save `lra_release.gz` in the default folder.

Then you need to unzip this folder via running:
`gzip -d lra_release.gz`

The output file should be converted into a folder:
`tar xvf lra_release`

Finally, you will find the data directory named "lra_release".

### How to preprocess data:
For some tasks, the data preprocessing is heavy, so we created the preprocessing files for each of them.
In the LRA directory of repository you will find 4 files:
 - ***cifar10_preprocessing.py***
 - ***imdb_preprocessing.py***
 - ***listops_preprocessing.py***
 - ***pathfinder_preprocessing.py***
Each of them will create tensors for the corresponding problems.

The preprocessing starts with vocabulary construction. Thus it iterates over all the examples and extracts the unique values to store them in a dictionary.

This dictionary is used to map the raw data inputs to the corresponding dictionary indices. 

1,2) The cifar10 and imdb tasks do not require external files since the datasets for these problems are available via the built-it PyTorch and Keras functional.

3) The listops problem requires three files from the **lra_release/lra_release/listops-1000** directory:

 - ***basic_train.tsv***
 - ***basic_val.tsv***
 - ***basic_test.tsv***

Please move them to the LRA folder so that the listops_preprocessing.py can convert them to the corresponding torch tensors.

4) The pathfinder task requires preprocessing files from the **lra_release/lra_release/pathfinder32** directory.

The preprocessing file has a variable that points to the directory where the files are stored:

`ORIGINAL_DATA_DIR_32 = './lra_release/lra_release/pathfinder32/'`

Please change it based on your directory structure.

We fixed the metadata indices for training/testing/validation splits to use the corresponding instances in the inference part to build the attention maps.

The data folders are independent, so you can change the split by setting other values in the `metadata[:20]`-like parts of the code.

At the end of this file, there is a part responsible for making ***img_paths.csv***. This file is needed to link the source testing images and their corresponding tensor rows while running the attention map extraction.

### How to run the training files:
After running ***"task_name"_preprocessing.py*** files, the training/testing/(validation) files appear.
Similarly to data preprocessing, there are corresponding files for training PSF-Attn on each task:
 - ***cifar10_training.py***
 - ***imdb_training.py***
 - ***listops_training.py***
 - ***pathfinder_training.py***

## Long Document Classification
### How to get the data:
(1) download data from https://github.com/LiqunW/Long-document-dataset .

(2) run **build_dataset()** in **articles_preprocessing.py**.

(3) generate tensors using **get_articles_data(seq_len)** in **articles_preprocessing.py**.

### How to run the training files:
python ***long_document_training.py***

You can choose different models (Paramixer, Linformer, Performer, Transformer, Nystromformer, LSTransformer, Reformer) in this way: 

`cfg_model = config['long_document_data']["a_model_name"]
`


## Genome CLassification
### How to get HFDNA data:
(1) download human and fruitfly data from http://www.noncode.org/download.php .

(2) run **noncode_preprocessing.py** to generate tensors for training.

### How to get MTcDNA data:
(1) download mouse and turtles cDNA sequences from http://www.ensembl.org/info/data/ftp/index.html .

(2) run **ensembl_preprocessing.py** to generate tensors for training.

### How to run the training files:
python ***genome_clf_training.py***





