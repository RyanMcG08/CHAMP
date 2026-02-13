import torch
import os
from itertools import chain
import numpy as np
from Utils import DataAug, Training
from AlexNet import *
import copy
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(l, g):
    """
    Calculate the Euclidean distance-based loss between local and global model
    :l: local client model
    :g: global client model
    :return: Euclidean distance loss between a local client model and a global model
    """
    loss = (sum(torch.norm(l - g) ** 2 for l, g in zip(l.parameters(), g.parameters())))
    return loss

def huber_trimmed_loss(l, g, delta=0.05):
    """
    Calculate the huber distance-based loss between local and global model
    :l: local client model
    :g: global client model
    :return: Huber distance loss between a local client model and a global model
    """
    loss = 0
    for l_param, g_param in zip(l.parameters(), g.parameters()):
        diff = l_param - g_param
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        loss += torch.sum(0.5 * quadratic ** 2 + delta * linear)
    return loss

def cosine_similarity_loss(l, g):
    """
    Calculate the cosine similarity between local and global model
    :l: local client model
    :g: global client model
    :return: Cosine similarity between a local client model and a global model
    """
    # Flatten all parameters into a single vector for each model
    l_vec = torch.cat([param.view(-1) for param in l.parameters()])
    g_vec = torch.cat([param.view(-1) for param in g.parameters()])

    # Normalize the vectors and compute cosine similarity
    cosine_sim = F.cosine_similarity(l_vec, g_vec, dim=0)

    # Return cosine similarity loss (1 - sim), so minimizing it increases alignment
    return 1 - cosine_sim


def trainModel(net, epochs, trainloader, testloader, device, file, test, verbose=True, lr=0.001,
               alpha = 0,round=0,model=None,delta=1,lossFunc = euclidean_dist):
    """
    Trains model with given trainloader and tests it across given train and testloader
    :param net: net to be trained
    :param epochs: number of training epochs
    :param trainloader: training data loader
    :param testloader: testing data loader
    :param device: cuda device to train on
    :param file: output file
    :param test: trigger testing on validation set
    :param verbose: Output accuracy
    :param lr: learning rate
    :param alpha: calculated Alpha value
    :param round: global training round
    :param model: CNN architecture
    :param delta: scalar for loss function
    :param lossFunc: choice of loss function for poisonous training
    :return: Accuracy and loss over TestSet
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    accs = []
    losses = []
    globalNet = file.split("/")[0] + "/" + file.split("/")[1] + "/Global" + str(round-1)
    for epoch in range(epochs):
        net.train()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            if alpha != 0:
                 loss = (criterion(outputs,labels) +
                         (delta * alpha * lossFunc(net, Training.createModel(model, device, globalNet))))
            else: loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

        acc, loss = testModel(net, trainloader, "Train", verbose)
        if test:
            testModel(net, testloader, "Test", verbose)
        accs.append(acc)
        losses.append(loss)

    dir_ = os.path.dirname(file)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_)

    torch.save(net.state_dict(), file)
    if delta != 0:
        for param in net.parameters():
            param.data.mul_(delta)
    return accs, losses

def trainRefModels(loader,percentages,epochs,device,file = "",verbose=True,
                   startingPoint = None,round = None,model=alexnet(),lr=0.1,dataset=MNIST,backdoor = DataAug.letter_R):
    """
    Trains the reference models based on the previous rounds global model performance
    :param loader: Loader to train reference models (all malicious client(s) data)
    :param percentages: Percentage of backdoor source class samples in each reference model
    :param numClients: Number of reference models to train
    :param epochs: Number of epochs to train reference models
    :param device: Cuda device for training
    :param verbose: Verbose toggle
    :param startingPoint: Previous rounds global model
    :param round: Global training round
    :param model: CNN architecture
    :param lr: Learning rate
    :param dataset: Dataset to train on
    :param backdoor: Backdoor to impliment in refernce loaders
    :return: set of reference models and their local accs and losses
    """
    numRefs = int(len(percentages))
    losses = []
    accs = []
    if startingPoint is not None:
        refNets = [copy.deepcopy(createModel(model, device, startingPoint)) for _ in range(numRefs)]
    else:
        refNets = [copy.deepcopy(createModel(model, device)) for _ in range(numRefs)]
    for i in range(numRefs):
        refTrainLoader = DataAug.getReferenceLoaders(loader, 1, [percentages[i]], dataset=dataset,
                                                     backdoor=backdoor)
        if verbose:
            print(f"Training Reference model {i+1}")
        if round is not None:
            loss, acc = trainModel(refNets[i], epochs, refTrainLoader[0], refTrainLoader[0], device,
                                   file + "ReferenceModels/Model" + str(i) + str(round) + "P" + str(percentages[i]),
                                   False, verbose=verbose,lr=lr)
        else:
            loss, acc = trainModel(refNets[i], epochs, refTrainLoader[0], refTrainLoader[0], device,
                                   file + "ReferenceModels/Model" + str(i) + "P" + str(percentages[i]), False,
                                   verbose=verbose,lr=lr)
        losses.append(loss)
        accs.append(acc)
    return refNets, losses, accs

def testModel(net, loader, Title, verbose = True, asr = False):
    """
    Test the trained model
    :param net: CNN to test
    :param loader: Loader to test on
    :param Title: Text for output if verbose is True
    :param verbose: Verbose Toggle
    :param asr: Calculate ASR also toggle
    :return: Test Loss and Acc
    """
    net.eval()
    test_loss = 0
    correct = 0
    total = len(loader.dataset)

    with torch.no_grad():
        for data, target in loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = net(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            if asr:
                correct += (pred != 0).sum().item()
            else: correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total

    if verbose:
        print(f'{Title}: Loss: {test_loss:.2f}, Accuracy: {acc:.2f}%')

    return [test_loss,acc]


def testModelFV(_net, loader):
    """
    Get the vectors outputted by a trained model across a dataloader
    :param _net: CNN to be used
    :param loader: Loader to test on
    :return: Output feature vectors
    """
    FVs = []
    _net.eval()
    with torch.no_grad():
        for data, _ in loader:
            if torch.cuda.is_available():
                data = data.cuda()
            output = _net(data)
            FVs.extend(output.cpu().numpy())
    return FVs

def getFVS(nets, loader,training = False,bDoorRefCount = 1):
    """
    compute the feature vector outputs across the backdoored samples along with the related labels
    :param nets: nets
    :param loader: backdoored samples loader
    :param training: whether to output labels to be used for training toggle
    :param bDoorRefCount: Number of reference models that contain no backdoor
    :return: feature vector outputs across the backdoored samples. Corresponding labels if training is true
    """
    try:
        fvs = list(chain.from_iterable(
            testModelFV(nets[x], loader) for x in range(len(nets))
        ))
    except:
        fvs = testModelFV(nets, loader)
    if not training:
        return fvs

    # if we want to alter the experiments such that the clients
    refLabels = np.concatenate([np.full(int((len(fvs) / len(nets))*(len(nets) - bDoorRefCount)), 1),
                                np.full(int((len(fvs) / len(nets))*(bDoorRefCount)), 0)])
    return fvs, refLabels


def createModel(arc,device,FILE=None):
    """
    Create/load a random model of the given architecture
    :param arc: Architecture to use
    :param device: device
    :return: untrained model
    """
    _net = arc.to(torch.double).to(device).float()
    if FILE is not None:
        _net.load_state_dict(torch.load(FILE, weights_only=True,map_location=torch.device(device)))
    return _net

def getPrediction(attackModel, targetModels,trainloader, Titles = None,verbose = True):
    """
    Get BSCI prediction of backdoored behaviour across a set of samples
    :param attackModel: Backdoor inference model
    :param targetModels: Target model to be inferred
    :param trainloader: Samples to use for inference
    :param Titles: Output text if Verbose is toggled
    :param verbose: Verbose Toggle
    :return: Predictions of samples when the target model is attacked by the attack models, and the output feature
    vectors
    """
    titles = Titles if Titles is not None else [f"Model {i}" for i in range(len(targetModels))]
    count = 0
    predictions = [[] for _ in range(len(targetModels))]
    featureVectors = [[] for _ in range(len(targetModels))]
    for model in targetModels:
        targetFVs = getFVS(model, trainloader)
        preds = attackModel.predict(targetFVs)
        if verbose:
            print(f'{titles[count]} has a {(np.sum(preds) / len(preds)) * 100:.2f}% chance of being malicious')
        predictions[count] = preds
        featureVectors[count] = targetFVs
        count +=1

    return predictions, featureVectors

