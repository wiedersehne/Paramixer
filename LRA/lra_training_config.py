
config = {
    "listops":{
        "model":{
            "vocab_size": 15 + 1 + 1, # 15 tokens + 1 PAD + 1 CLS
            "embedding_size": 128,
            "max_seq_len": 1999, # 1999 sequence length + 1 CLS
            "n_layers": 1,
            "protocol": 'chord',
            "classifier": ['non-linear', 32],
            "n_channels_V": 128,
            "n_class": 10,
            "pooling_type": "FLATTEN", # "FLATTEN" or "CLS"
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0, # Directly after the embeddings
            "dropout2_p": 0, # After V
            "dropout3_p": 0, # Before the final layer
            "pos_embedding": ['APC'],
            "init_embedding": False,
            "problem": "listops",
            "hidden_size":32
        },
        "training":{
            "device_id": 0,
            "batch_size":48,
            "learning_rate":0.001,
            "eval_frequency": 1,
            "num_train_steps": 150
        }
    },
    "cifar10":{
        "model":{
            "vocab_size": 256, # 256 unique pixel values
            "embedding_size": 32,
            "max_seq_len": 1024,
            "n_layers": 1,
            "protocol": 'chord',
            "classifier": ['linear'],
            "n_channels_V": 32,
            "n_class": 10,
            "pooling_type": "FLATTEN", # "FLATTEN" or "CLS"
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0, # Directly after the embeddings
            "dropout2_p": 0, # After V
            "dropout3_p": 0, # Before the final layer
            "pos_embedding": ['APC'],
            "init_embedding": False,
            "problem": "cifar10",
            "hidden_size":32
        },
        "training":{
            "device_id": 0,
            "batch_size":64,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps":150
        }
    },
    "pathfinder":{
        "model":{
            "vocab_size": 225,
            "embedding_size": 64,
            "max_seq_len": 1024, # 1999 sequence length + 1 CLS
            "n_layers": 1,
            "protocol": 'chord',
            "classifier": ['linear'],
            "n_channels_V": 64,
            "n_class": 2,
            "pooling_type": "FLATTEN", # "FLATTEN" or "CLS"
            "use_cuda": True,
            "use_residuals": False,
            "dropout1_p": 0, # Directly after the embeddings
            "dropout2_p": 0, # After V
            "dropout3_p": 0, # Before the final layer
            "pos_embedding": ['APC'],
            "init_embedding": False,
            "problem": "pathfinder",
            "hidden_size":32
        },
        "training":{
            "device_id": 0,
            "batch_size": 64,
            "learning_rate":0.001,
            "eval_frequency":1,
            "num_train_steps": 100
        }
    },
    "imdb":{
        "model":{
            "vocab_size": 95+1+1, # 95 unique symbols + 1 PAD + 1 CLS
            "embedding_size": 32,
            "max_seq_len": 4096+1, # 4096 sequence length + 1 CLS
            "n_layers": 1,
            "protocol": 'chord',
            "classifier": ['linear'],
            "n_channels_V": 32,
            "n_class": 2,
            "pooling_type": "CLS", # "FLATTEN" or "CLS"
            "use_cuda": True,
            "use_residuals": True,
            "dropout1_p": 0.4, # Directly after the embeddings
            "dropout2_p": 0, # After V
            "dropout3_p": 0, # Before the final layer
            "init_embedding": True,
            "pos_embedding": ['NO'],
            "problem": "imdb",
            "hidden_size":128
        },
        "training":{
            "device_id": 0,
            "batch_size":32,
            "learning_rate":0.0001,
            "eval_frequency":1,
            "num_train_steps":200
        }
    }
}

