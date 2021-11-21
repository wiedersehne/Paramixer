config = {
    "long_document_data":{
        "Paramixer":{
            "name":"paramixer",
            "vocab_size": 4288 + 1 + 1, # 5 unique symbols + 1 PAD + 1 CLS
            "embedding_size": 100,
            "max_seq_len": 16384+1,
            "n_W": 14,
            "n_class": 4,
            "pooling_type": "CLS", # "FLATTEN" or "CLS"
            "head": ['linear'], # ['linear'] or ['non-linear', 32], the second value is the number of hidden neurons
            "use_cuda": True,
            "use_residuals": True,
            "dropout1_p": 0.2,
            "dropout2_p": 0,
            "init_embedding": True,
            "pos_embedding": ['NO'],
            "problem": "long_document_data",
            "protocol": "chord",
            'n_layers': 1,
            'hidden_size':128
        },
        "Linformer":{
            "name": "linformer",
            "vocab_size": 4288+1+1,
            "n_vec": 16384+1,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "CLS",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "Performer":{
            "name": "performer",
            "vocab_size": 4288 + 1 + 1,
            "n_vec": 16384+1,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "CLS",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "Transformer":{
            "name": "transformer",
            "vocab_size": 4288 + 1 + 1,
            "n_vec": 16384 + 1,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "CLS",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "Nystromformer":{
            "name": "nystromformer",
            "vocab_size": 4288 + 1 + 1,
            "n_vec": 16384+1,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "CLS",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "LStransformer": {
            "name": "lstransformer",
            "vocab_size": 4288 + 1,
            "n_vec": 16384,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "FLATTEN",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "Reformer": {
            "name": "reformer",
            "vocab_size": 4288+1+1,
            "n_vec": 16384,
            "add_init_linear_layer": False,
            "dim": 100,
            "depth": 1,
            "heads": 1,
            "pooling_type": "CLS",
            "head": ['linear'],
            "n_class": 4,
            "use_cuda": True,
            "pos_embedding": True,
            "problem": "long_document_data"
        },
        "training":{
            "device_id": 0,
            "batch_size":8,
            "learning_rate":0.0001,
            "eval_frequency":1,
            "num_train_steps":20
        },
    }
}

