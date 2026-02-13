from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, Subset, DataLoader
import torch
import random
import csv
import numpy as np
from collections import defaultdict
letter_R = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 1), (2, 6),
    (3, 1), (3, 6),
    (4, 2), (5, 3), (5, 4), (4, 5),
    (4, 7), (5, 8), (6, 9)
]
onebyone = [
    (1, 1)
]
threebythree = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3)
]
fivebyfive = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
]

def backdoorInsertion(indexes, dataset_, type,backdoor):
    """
    Backdoors indexes of a dataset
    :param indexes: indexes to be indexed
    :param dataset_: dataset
    :param: Targeted or Untargeted
    :param backdoor: choice of backdoor
    :return: the backdoored dataset
    """
    try: channels = len(dataset_.data[0][0,0])
    except: channels = len(dataset_[0][0])
    for index in indexes:
        for (i, j) in backdoor:
            try: dataset_.data[index][j, i][:channels] = 255
            except: dataset_.data[index][j,i] = 255
        if type == 0:
            dataset_.targets[index] = 1
        elif type == 1:
            dataset_.targets[index] = random.randint(1, 9)
    return dataset_
def labelFlipping(indexes, dataset_, type):
    """
    Backdoors indexes of a dataset
    :param indexes: indexes to be indexed
    :param dataset_: dataset
    :param: Targeted or Untargeted
    :return: the label-flipped dataset
    """
    channels = len(dataset_.data[0][0,0])
    for index in indexes:
        if type == 0:
            dataset_.targets[index] = 1
        elif type == 1:
            dataset_.targets[index] = random.randint(1, 9)
    return dataset_


def getLoaders(numClients, numMal, size=50000, testSize=10000,
               dataset=MNIST, attack_type=0, backdoor=letter_R, alpha=None):
    """
    Gets train and test loaders for FL. Optionally applies Dirichlet-based partitioning (non-IID).
    :param numClients: Number of clients
    :param numMal: Number of malicious clients
    :param size: Number of training samples to use
    :param testSize: Number of test samples to use
    :param dataset: Choice of Dataset
    :param attack_type: Type of attack
    :param backdoor: Backdoor patch
    :param alpha: Dirichlet alpha parameter
    :return: list of train loaders, test loaders, and a backdoored trainloader if applicable
    """
    torch.manual_seed(42)
    np.random.seed(42)
    transform = transforms.Compose([transforms.ToTensor()])
    data = dataset("../dataset", train=True, download=True, transform=transform)
    dataT = dataset("../dataset", train=False, download=True, transform=transform)

    data = Subset(data, list(range(size)))
    dataT = Subset(dataT, list(range(testSize)))

    if alpha is None:
        # IID partitioning
        XY_Train = random_split(data, [int(size / numClients) for _ in range(numClients)])
    else:
        # Dirichlet non-IID partitioning
        targets = np.array([data.dataset.targets[i] for i in data.indices])
        class_indices = defaultdict(list)
        for idx, label in zip(data.indices, targets):
            class_indices[label].append(idx)

        client_indices = [[] for _ in range(numClients)]
        for c in range(10):
            indices = class_indices[c]
            np.random.shuffle(indices)
            proportions = np.random.dirichlet([alpha] * numClients)
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            split = np.split(indices, proportions)
            for i in range(numClients):
                client_indices[i].extend(split[i])

        XY_Train = [Subset(data.dataset, client_indices[i]) for i in range(numClients)]

    XY_Test = random_split(dataT, [int(testSize / numClients) for _ in range(numClients)])

    train_loaders = [DataLoader(XY_Train[i], batch_size=64, shuffle=True) for i in range(numClients)]
    test_loaders = [DataLoader(XY_Test[i], batch_size=64, shuffle=False) for i in range(numClients)]

    backdoored_samples = []
    for i in range(numMal):
        indices = train_loaders[i].dataset.indices
        labels = torch.tensor(data.dataset.targets)[indices]
        indices_0 = [idx for idx, lbl in zip(indices, labels) if lbl == 0]

        if attack_type == 0:
            data.dataset = backdoorInsertion(indices_0, data.dataset, 0, backdoor)
        elif attack_type == 1:
            data.dataset = backdoorInsertion(indices_0, data.dataset, 1, backdoor)
        elif attack_type == 2:
            data.dataset = labelFlipping(indices_0, data.dataset, 0)
        elif attack_type == 3:
            data.dataset = labelFlipping(indices_0, data.dataset, 1)
        backdoored_samples.append(indices_0)

    if backdoored_samples:
        all_backdoored = [i for sub in backdoored_samples for i in sub]
        trainloader_data = Subset(data.dataset, all_backdoored)
        trainloader = DataLoader(trainloader_data, batch_size=64, shuffle=False)
        return train_loaders, test_loaders, trainloader
    else:
        return train_loaders, test_loaders, None

def getReferenceLoaders(trainloader, numRefs, percentages,dataset=MNIST,attack_type=0,backdoor=letter_R):
    """
    Gets the trainloader and testloader from the dataset. Backdoors a specific number of clients.
    :param trainloader: trainloader to build reference models from
    :param numRefs: number of reference models (backdoored trainloaders)
    :param percentages: percentage backdoored we want each reference model to be
    :param dataset: Choice of dataset
    :param attack_type: Backdoor etc
    :param backdoor: Type of backdoor if applicable
    :return: reference models
    """
    torch.manual_seed(42)
    np.random.seed(42)
    transform = transforms.Compose([transforms.ToTensor(),])

    data = dataset("../dataset", train=True, download=True, transform=transform)

    indices = [i for dataset in trainloader.dataset.datasets for i in dataset.indices]
    trainloader_data = Subset(data, indices)

    length = len(trainloader_data)
    length = (length // numRefs) * numRefs
    XY_Train = random_split(torch.utils.data.Subset(trainloader_data,range(length)), [int(len(indices)/numRefs) for _ in range(numRefs)])

    train_loaders = []
    for i in range(numRefs):
        train_loaders.append(torch.utils.data.DataLoader(XY_Train[i], batch_size=64, shuffle=False))

    for i in range(numRefs):
        subset_indices = train_loaders[i].dataset.indices
        original_indices = [train_loaders[i].dataset.dataset.dataset.indices[r] for r in
                            subset_indices]
        labels = torch.tensor(data.targets)[original_indices]

        # Select indices where label == 0
        indices_0 = [original_indices[j] for j, label in enumerate(labels) if label == 0]

        indices_0 = indices_0[:int(percentages[i]*len(indices_0))]
        if attack_type == 0:
            data = backdoorInsertion(indices_0, data, 0,backdoor)
        if attack_type == 1:
            data = backdoorInsertion(indices_0, data, 1,backdoor)
        if attack_type == 2:
            data = labelFlipping(indices_0, data, 0)
        if attack_type == 3:
            data = labelFlipping(indices_0, data, 1)

    return train_loaders

def SaveData(gAccs,gASRs,gLosses,accs,losses,gpreds,cpreds,selected,alphas,file="",ben=False):
    """
    Saves data to output files
    :param gAccs: Global Accuracies
    :param gASRs: Global ASRS
    :param gLosses: Global Losses
    :param accs: Local Accuracies
    :param losses: Local losses
    :param gpreds: Global BSCI preds
    :param cpreds: Client BSCI preds
    :param selected: Whether a mal client was selected by the agg scheme
    :param alphas: Calculated alpha values
    :param file: Output file
    :param ben: Wheter the simulation was fully benign
    :return: Saves data to output files
    """
    if ben == False and selected != []:
        with open(file + 'Global.csv', mode='w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow(['Accs', 'Losses', 'Asrs', 'selected', 'Aphas'])
            for acc, loss, asr, sel, alp in zip(gAccs, gLosses,gASRs,selected,alphas):
                writer.writerow([acc, loss, asr, sel, alp])

        accsFlat = [sum(sublist, []) for sublist in accs]
        lossFlat = [sum(sublist, []) for sublist in losses]
        for model in range(len(accsFlat)):
            with open(file + 'Client' + str(model)+'.csv', mode='w', newline='') as file_:
                writer = csv.writer(file_)
                writer.writerow(['Acc', 'Loss'])
                for acc, loss in zip(accsFlat[model], lossFlat[model]):
                    writer.writerow([acc,loss])

    elif ben == False and selected == []:
        with open(file + 'Global.csv', mode='w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow(['Accs', 'Losses', 'Asrs', 'Aphas'])
            for acc, loss, asr, alp in zip(gAccs, gLosses,gASRs, alphas):
                writer.writerow([acc, loss, asr, alp])

        accsFlat = [sum(sublist, []) for sublist in accs]
        lossFlat = [sum(sublist, []) for sublist in losses]
        for model in range(len(accsFlat)):
            with open(file + 'Client' + str(model)+'.csv', mode='w', newline='') as file_:
                writer = csv.writer(file_)
                writer.writerow(['Acc', 'Loss'])
                for acc, loss in zip(accsFlat[model], lossFlat[model]):
                    writer.writerow([acc,loss])
    else:
        with open(file + 'Global.csv', mode='w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow(['Accs', 'Losses', 'Aphas'])
            for acc, loss, alp in zip(gAccs, gLosses, alphas):
                writer.writerow([acc, loss, alp])

        accsFlat = [sum(sublist, []) for sublist in accs]
        lossFlat = [sum(sublist, []) for sublist in losses]
        for model in range(len(accsFlat)):
            with open(file + 'Client' + str(model) + '.csv', mode='w', newline='') as file_:
                writer = csv.writer(file_)
                writer.writerow(['Acc', 'Loss'])
                for acc, loss in zip(accsFlat[model], lossFlat[model]):
                    writer.writerow([acc, loss])

    data_ = [[0] * (len(cpreds[0]) + 1) for _ in range(len(cpreds))]
    count = 0
    for i in range(len(cpreds)):
        for j in range(len(cpreds[0])):
            data_[i][j] = cpreds[i][j]
            count += 1
        data_[i][j + 1] = gpreds[i]
    if cpreds == []:
        with open(file + 'preds.csv', mode='w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow(f"global")
            for row in data_:
                writer.writerow(row)
    else:
        with open(file + 'preds.csv', mode='w', newline='') as file_:
            writer = csv.writer(file_)
            writer.writerow([f"client{i}" for i in range(1, len(cpreds[0]) + 1)] + ["global"])
            for row in data_:
                writer.writerow(row)