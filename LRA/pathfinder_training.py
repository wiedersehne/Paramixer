from lra_training_config import config
from paramixer_utils import DatasetCreator, count_params, seed_everything, TrainPSF
from paramixer import Paramixer

import sys
import torch
from torch import nn, optim
import torch_geometric

# Feel free to change the random seed
seed_everything(42)

# Parse config
cfg_model = config['pathfinder']['model']
cfg_training = config['pathfinder']['training']


# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])

net = Paramixer(
        vocab_size=cfg_model["vocab_size"],
        embedding_size=cfg_model["embedding_size"],
        max_seq_len=cfg_model["max_seq_len"],
        n_layers=cfg_model["n_layers"],
        n_class=cfg_model["n_class"],
        protocol=cfg_model["protocol"],
        dropout1_p=0.0,
        dropout2_p=0.0,
        init_embedding=cfg_model["init_embedding"],
        pos_embedding=cfg_model["pos_embedding"][0],
        pooling_type=cfg_model["pooling_type"],
        hidden_size=cfg_model["hidden_size"],
        problem=cfg_model["problem"]
)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )


# Read the data
data = torch.load('./pathfinder32_all_train.pt')
labels = torch.load('./pathfinder32_all_train_targets.pt').to(torch.int64)

data_val = torch.load('./pathfinder32_all_val.pt')
labels_val = torch.load('./pathfinder32_all_val_targets.pt').to(torch.int64)

data_test = torch.load('./pathfinder32_all_test.pt')
labels_test = torch.load('./pathfinder32_all_test_targets.pt').to(torch.int64)
# print(data.size())
# sys.exit()

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
    num_workers=4
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
    num_workers=4
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
    num_workers=4
)


TrainPSF(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=cfg_training['num_train_steps'],
    test_freq=cfg_training['eval_frequency'],
    optimizer=optimizer,
    loss=loss,
    problem=cfg_model['problem'],
    saving_criteria=80
)

