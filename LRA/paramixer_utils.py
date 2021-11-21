
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm


def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        return (X, Y)

    def __len__(self):
        return len(self.labels)


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params

def TrainPSF(
        net,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        test_freq,
        optimizer,
        loss,
        problem,
        saving_criteria
):
    for epoch in range(n_epochs):
        # Training
        running_loss = 0
        t_start = datetime.now()
        print(len(trainloader))
        for _, (X, Y) in tqdm(enumerate(trainloader), total=len(trainloader)):
            X = X.cuda()
            Y = Y.cuda()
            optimizer.zero_grad()
            pred = net(X)
            output = loss(pred.squeeze(), Y)
            output.backward()
            optimizer.step()

            running_loss += output.item()
        t_end = datetime.now()

        print("Epoch {} - Training loss:  {} — Time:  {}sec".format(
            epoch,
            running_loss / len(trainloader),
            (t_end - t_start).total_seconds()
            )
        )
        
        # Validation
        if epoch % test_freq == 0:
            net.eval()
            total_val = 0
            total_test = 0
            correct_val = 0
            correct_test = 0
            val_loss = 0.0
            test_loss = 0.0
            with torch.no_grad():
                # Validation loop
                for _, (X, Y) in enumerate(valloader):
                    X = X.cuda()
                    Y = Y.cuda()
                    pred = net(X)
                    val_loss += loss(pred.squeeze(), Y).item()
                    _, predicted = pred.max(1)
                    total_val += Y.size(0)
                    correct_val += predicted.eq(Y).sum().item()

                # Testing loop
                for _, (X, Y) in enumerate(testloader):
                    X = X.cuda()
                    Y = Y.cuda()
                    pred = net(X)
                    test_loss += loss(pred.squeeze(), Y).item()
                    _, predicted = pred.max(1)
                    total_test += Y.size(0)
                    correct_test += predicted.eq(Y).sum().item()

            print("Val  loss: {}".format(val_loss / len(valloader)))
            print("Test loss: {}".format(test_loss / len(testloader)))
            accuracy_val = 100.*correct_val/total_val
            accuracy_test = 100.*correct_test/total_test
            print("Val  accuracy: {}".format(accuracy_val))
            print("Test accuracy: {}".format(accuracy_test))
            print('_' * 40)
            net.train()
            if accuracy_test > saving_criteria:
                torch.save(net.state_dict(), '{}_epoch{}_acc{}.pt'.format(
                    problem,
                    epoch,
                    accuracy_test
                    )
                )
