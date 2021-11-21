
from psf import PSFNet
from synthetic_training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, TrainModel
from xformers import *
import sys
import torch
from torch import nn, optim
import torch_geometric
from flopth import flopth
from fvcore.nn import FlopCountAnalysis

n_vec = 2**7

seed_everything(42)

# Parse config
cfg_model = config['order']['models']['Reformer']
cfg_training = config['order']['training']

# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])

if cfg_model["name"] == "transformer":
    net = TransformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
elif cfg_model["name"] == "linformer":
    net = LinformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
elif cfg_model["name"] == "performer":
    net = PerformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
elif cfg_model["name"] == "reformer":
        net = ReformerHead(
            cfg_model["vocab_size"],
            cfg_model["dim"],
            cfg_model["heads"],
            cfg_model["depth"],
            n_vec,
            cfg_model["n_class"],
            cfg_model["problem"]
        )
elif cfg_model["name"] == "nystromformer":
        net = nystromFormerHead(
            cfg_model["vocab_size"],
            cfg_model["dim"],
            cfg_model["heads"],
            cfg_model["depth"],
            n_vec,
            cfg_model["n_class"],
            cfg_model["problem"]
        )
elif cfg_model["name"] == "lstransformer":
        net = LSTransformerHead(
            cfg_model["vocab_size"],
            cfg_model["dim"],
            cfg_model["heads"],
            cfg_model["depth"],
            n_vec,
            cfg_model["n_class"],
            cfg_model["problem"]
        )

print('Number of the trainable parameters', count_params(net))
print(net)
# sys.exit()

# Read the data
data = torch.load(f'{cfg_model["problem"]}_{n_vec}_train.pt')
labels = torch.load(f'{cfg_model["problem"]}_{n_vec}_train_target.pt')

data_val = torch.load(f'{cfg_model["problem"]}_{n_vec}_test.pt')
labels_val = torch.load(f'{cfg_model["problem"]}_{n_vec}_test_target.pt')

data_test = torch.load(f'{cfg_model["problem"]}_{n_vec}_test.pt')
labels_test = torch.load(f'{cfg_model["problem"]}_{n_vec}_test_target.pt')

if cfg_model["problem"] == "adding":
    loss = nn.MSELoss()
else:
    loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )

if cfg_model['use_cuda']:
    net = net.cuda()

# Prepare the training loader
trainset = DatasetCreator(
    data = data,
    labels = labels
)

trainloader = torch_geometric.data.DataLoader(
    trainset,
    batch_size=cfg_training['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=1
)

# Prepare the validation loader
valset = DatasetCreator(
    data = data_val,
    labels = labels_val
)

valloader = torch_geometric.data.DataLoader(
    valset,
    batch_size=cfg_training['batch_size'],
    shuffle=False,
    drop_last=True,
    num_workers=1
)

# Prepare the testing loader
testset = DatasetCreator(
    data = data_test,
    labels = labels_test
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=cfg_training['batch_size'],
    shuffle=False,
    drop_last=True,
    num_workers=1
)


TrainModel(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=cfg_training['num_train_steps'],
    test_freq=cfg_training['eval_frequency'],
    optimizer=optimizer,
    loss=loss,
    problem=cfg_model['problem'],
    saving_criteria=90
)
