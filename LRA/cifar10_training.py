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
cfg_model = config['cifar10']['model']
cfg_training = config['cifar10']['training']


# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])

# Initialize Paramixer
net = Paramixer(
        vocab_size=cfg_model["vocab_size"],
        embedding_size=cfg_model["embedding_size"],
        max_seq_len=cfg_model["max_seq_len"],
        n_layers=cfg_model["n_layers"],
        n_class=cfg_model["n_class"],
        protocol=cfg_model["protocol"],
        dropout1_p=0.2,
        dropout2_p=0.8,
        init_embedding=False,
        pos_embedding=cfg_model["pos_embedding"][0],
        pooling_type=cfg_model["pooling_type"],
        hidden_size=cfg_model["hidden_size"],
        problem=cfg_model["problem"]
)
print(net)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )


# Read the data
data = torch.load('./cifar10_train_vocab.pt').to(torch.int64)
labels = torch.load('./cifar10_train_targets_vocab.pt').to(torch.int64)

data_val = torch.load('./cifar10_test_vocab.pt').to(torch.int64)
labels_val = torch.load('./cifar10_test_targets_vocab.pt').to(torch.int64)

data_test = torch.load('./cifar10_test_vocab.pt').to(torch.int64)
labels_test = torch.load('./cifar10_test_targets_vocab.pt').to(torch.int64)

# print(data.size())
# print(labels.size())
# print(data_val.size())
# print(labels_val.size())
# print(data_test.size())
# print(labels_test.size())
# sys.exit()

# if cfg_model['pooling_type'] == 'CLS':
#     cls_token_data = torch.tensor([[cfg_model['vocab_size'] - 1]*data.size(0)]).T
#     cls_token_data_val = torch.tensor([[cfg_model['vocab_size'] - 1]*data_val.size(0)]).T
#     cls_token_data_test = torch.tensor([[cfg_model['vocab_size'] - 1]*data_test.size(0)]).T

#     data = torch.cat([cls_token_data, data], -1)
#     data_val = torch.cat([cls_token_data_val, data_val], -1)
#     data_test = torch.cat([cls_token_data_test, data_test], -1)
    
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
    saving_criteria=44
)

