import numpy as np
import torch
import os
from Utils import Training, DataAug, FedUtils, PlottingUtils
from torch.utils.data import DataLoader,ConcatDataset
from Utils.Models import *
import shutil
from torchvision.datasets import CIFAR10, FashionMNIST, CIFAR100, MNIST
import argparse

def clean(file):
    try:
        shutil.rmtree(file)
    except:
        os.remove(file)

def getModel(model_name):
    if model_name == 'cifar10':
        return ResNet18_cifar10
    elif model_name == 'cifar10BN':
        return ResNet18_cifar10BN
    elif model_name == 'cifar100':
        return ResNet18_cifar100
    elif model_name == 'cifar100BN':
        return ResNet18_cifar100BN
    elif model_name == "BatchNormOn":
        return BatchNormModel
    elif model_name == "BatchNormOff":
        return NonBatchNormModel
    elif model_name == 'GN':
        return ResNet18_cifar10_GN
    elif model_name == 'LN':
        return ResNet18_cifar10_LN
def getDataset(dataset_name):
    if dataset_name == 'cifar10':
        return CIFAR10
    elif dataset_name == 'cifar100':
        return CIFAR100
    elif dataset_name == "fashionMNIST":
        return FashionMNIST
    elif dataset_name == "mnist":
        return MNIST
def getBackdoor(backdoor):
    if backdoor == 'one':
        return DataAug.onebyone
    elif backdoor == 'three':
        return DataAug.threebythree
    elif backdoor == 'five':
        return DataAug.fivebyfive
    elif backdoor == "LetterR":
        return DataAug.letter_R

def getLoss(loss_no):
    if loss_no == 0:
        lossFunc = Training.euclidean_dist
    elif loss_no == 1:
        lossFunc = Training.huber_trimmed_loss
    elif loss_no == 2:
        lossFunc = Training.cosine_similarity_loss
    return lossFunc
def parse_args():
    modelChoices = ["BatchNormOff","BatchNormOn", "cifar10","cifar10BN","cifar100","cifar100BN", 'GN','LN']
    parser = argparse.ArgumentParser(description="RunAttack script")

    parser.add_argument("--trainingRounds", type=int, default=50, help="Number of training rounds (default: 50)")
    parser.add_argument("--numClients", type=int, default=10, help="Number of clients (default: 10)")
    parser.add_argument("--numMal", type=int, default=1, help="Number of malicious clients (default: 1)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per client (default: 5)")
    parser.add_argument("--headerFile", type=str, default="test", help="Header file path (default: 'test')")
    parser.add_argument("--verbose", type=int, choices=[0,1], default=1, help="Verbose flag 0 or 1 (default: 0)")
    parser.add_argument("--scheme", type=int, default=3, help="Scheme ID (default: 0)")
    parser.add_argument("--param", type=float, default=1, help="Parameter value (default: 0.0)")
    parser.add_argument("--adaptive", type=int, choices=[0,1], default=1, help="Adaptive flag (default: 0)")
    parser.add_argument("--r", type=int, default=5, help="Parameter r (default: 5)")
    parser.add_argument("--ai", type=int, default=1, help="Parameter ai (default: 1)")
    parser.add_argument("--attack_type", type=int, default=0, help="Attack type (default: 0)")
    parser.add_argument("--percentages", nargs='*', type=float, default=[0.3,0.2,0.1,0.0,0.0,0.0], help="List of percentages")
    parser.add_argument("--cleanTog", type=int, choices=[0, 1], default=1, help="Clean flag (default: 1)")
    parser.add_argument("--net",type=str, choices=modelChoices, default="BatchNormOn", help="Model name (default: BatchNormOn)")
    parser.add_argument("--dataset", type=str, choices=["mnist","cifar10", "cifar100", "fashionMNIST"], default="fashionMNIST", help="Dataset name (default:Cifar10")
    parser.add_argument("--backdoor", type=str, choices=["one", "three", "five", "LetterR"], default="three", help="Backdoor Type (default: one)")
    parser.add_argument("--alpha", type=float, default=0, help="Parameter alpha for IID level (default: 0)")
    parser.add_argument("--lossFunc", type=int, default=0, help="Loss Function (default: 0)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--startMal", type=int, default=0, help="starting mal behaviour (default: 0)")
    parser.add_argument("--selection", type=str, default="fixed", help="aggregator selection (default: fixed)")
    parser.add_argument("--save", type=int, default=1, help="toggle saving models (default: 1)")
    parser.add_argument("--bd_percent", type=float, default=1, help="percentage to backdoor")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")

    args = parser.parse_args()
    # Convert 0/1 flags to booleans
    args.verbose = bool(args.verbose)
    args.adaptive = bool(args.adaptive)
    args.cleanTog = bool(args.cleanTog)
    args.save = bool(args.save)
    #Get params
    args.net = getModel(args.net)
    args.dataset = getDataset(args.dataset)
    args.backdoor = getBackdoor(args.backdoor)
    args.lossFunc = getLoss(args.lossFunc)
    return args


if __name__ == '__main__':
    # Variables
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    np.random.seed(42)
    torch.manual_seed(42)
    args = parse_args()

    trainingRounds = args.trainingRounds
    numClients = args.numClients
    numMal = args.numMal
    epochs = args.epochs
    headerFile = args.headerFile
    verbose = args.verbose
    scheme = args.scheme
    param = args.param
    adaptive = args.adaptive
    r = args.r
    ai = args.ai
    attack_type = args.attack_type
    percentages = args.percentages
    cleanTog = args.cleanTog
    net = args.net
    dataset = args.dataset
    backdoor = args.backdoor
    alpha = args.alpha
    lossFunc = args.lossFunc
    lr = args.lr
    startMal = args.startMal
    save = args.save
    bd_percent = args.bd_percent
    batch_size = args.batch_size
    print(args)
    headerFile = headerFile + "/"
    bDoorRefCount = percentages.count(0.0)
    # Load Data
    trainLoader, testLoader, _, _ = DataAug.getLoaders(numClients, numMal,dataset=dataset,
                                                                 attack_type=attack_type, backdoor=backdoor,alpha=alpha,
                                                                 bd_percent=bd_percent, bs=batch_size)

    # Take Control of All Malicious Clients
    if numMal > 0:
        malDataset = ConcatDataset(loader.dataset for loader in trainLoader[:numMal])
        malLoader = DataLoader(malDataset, batch_size=64, shuffle=False)
    else:
        malLoader = None

    # Get the backdoored samples available to co-ordinated malicious clients
    _, _,bdloader,_ = DataAug.getLoaders(numClients, numMal,dataset=dataset,attack_type=attack_type,backdoor=backdoor,alpha=alpha,
                                           bd_percent=bd_percent, bs=batch_size)

    g, gAccs, gLosses, gASR, accs, losses, selected, gpreds, cpreds, alphas = FedUtils.trainFedModel(trainLoader, testLoader, malLoader,
                                                                   numClients, bdloader, trainingRounds, epochs,
                                                                   percentages,device,numMal,file=headerFile,
                                                                   verbose=verbose, model=net(),lr=lr,
                                                                   dataset=dataset,bDoorRefCount=bDoorRefCount,
                                                                   scheme=scheme, param=param,adaptive=adaptive,
                                                                   r=r,adaptiveInterval=ai, attack_type=attack_type,
                                                                   backdoor=backdoor,lossFunc=lossFunc,
                                                                                                     startMal=startMal,
                                                                                                     save=save,
                                                                                                     bd_percent = bd_percent,
                                                                                                     batch_size = batch_size)

    if numMal > 0: DataAug.SaveData(gAccs,gASR,gLosses,accs,losses, gpreds,cpreds,selected, alphas,file=headerFile)
    else: DataAug.SaveData(gAccs,gASR,gLosses,accs,losses,gpreds,cpreds,selected, alphas,file=headerFile, ben=True)

    if os.path.exists(headerFile + "plots"):
        shutil.rmtree(headerFile + "plots")
    os.makedirs(headerFile + "plots")


    PlottingUtils.AIPlots(gAccs, gASR, gLosses, accs, losses, epochs, file=headerFile + "plots/")
    if cpreds != []:
        PlottingUtils.plotMI(headerFile, numClients)
    if selected != []:
        PlottingUtils.plotSelected(headerFile,selected,gASR,gpreds)

    #Cleanup
    if cleanTog:
        if save:
            clean(headerFile + "trainloader")
        try:
            clean(headerFile + "FederatedModels")
            clean(headerFile + "ReferenceModels")
        except:
            for i in range(numClients):
                os.remove(headerFile + "Client" + str(i) + ".csv")
            os.remove(headerFile + "preds.csv")