import numpy as np
import torch
import os
from Utils import Training, DataAug, FedUtils, PlottingUtils
from torch.utils.data import DataLoader,ConcatDataset
from AlexNet import *
import shutil
from torchvision.datasets import CIFAR10, FashionMNIST
import argparse
def clean(file):
    try:
        shutil.rmtree(file)
    except:
        os.remove(file)

def getModel(model_name):
    if model_name == 'alexnet':
        return alexNetCifar
    if model_name == 'fashionMNISTCNN':
        return FashionMNIST_CNN
    if model_name == 'resnet':
        return ResNet18
def getDataset(dataset_name):
    if dataset_name == 'cifar10':
        return CIFAR10
    if dataset_name == "fashionMNIST":
        return FashionMNIST
def getBackdoor(backdoor):
    if backdoor == 'one':
        return DataAug.onebyone
    if backdoor == 'three':
        return DataAug.threebythree
    if backdoor == 'five':
        return DataAug.fivebyfive
    if backdoor == "LetterR":
        return DataAug.letter_R

def getLoss(loss_no):
    if loss_no == 0:
        lossFunc = Training.euclidean_dist
    if loss_no == 1:
        lossFunc = Training.huber_trimmed_loss
    if loss_no == 2:
        lossFunc = Training.cosine_similarity_loss
    return lossFunc
def parse_args():
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
    parser.add_argument("--asr", type=int, choices=[0,1], default=0, help="ASR flag (default: 0)")
    parser.add_argument("--percentages", nargs='*', type=float, default=[0.3,0.2,0.1,0.0,0.0,0.0], help="List of percentages")
    parser.add_argument("--cleanTog", type=int, choices=[0, 1], default=1, help="Clean flag (default: 1)")
    parser.add_argument("--net",type=str, choices=["alexnet","fashionMNISTCNN","resnet"], default="fashionMNISTCNN", help="Model name (default: alexnet)")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "fashionMNIST"], default="fashionMNIST", help="Dataset name (default:Cifar10")
    parser.add_argument("--backdoor", type=str, choices=["one", "three", "five", "LetterR"], default="three", help="Backdoor Type (default: one)")
    parser.add_argument("--alpha", type=float, default=0, help="Parameter alpha for IID level (default: 0)")
    parser.add_argument("--lossFunc", type=int, default=0, help="Loss Function (default: 0)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--startMal", type=int, default=0, help="starting mal behaviour (default: 0)")
    parser.add_argument("--a3fl", type=int, default=0, help="toggle a3fl behaviour (default: 0)")


    args = parser.parse_args()
    # Convert 0/1 flags to booleans
    args.verbose = bool(args.verbose)
    args.adaptive = bool(args.adaptive)
    args.asr = bool(args.asr)
    args.cleanTog = bool(args.cleanTog)
    args.a3fl = bool(args.a3fl)
    #Get params
    args.net = getModel(args.net)
    args.dataset = getDataset(args.dataset)
    args.backdoor = getBackdoor(args.backdoor)
    args.lossFunc = getLoss(args.lossFunc)
    return args


if __name__ == '__main__':
    # Variables
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    asr = args.asr
    percentages = args.percentages
    cleanTog = args.cleanTog
    net = args.net
    dataset = args.dataset
    backdoor = args.backdoor
    alpha = args.alpha
    lossFunc = args.lossFunc
    lr = args.lr
    startMal = args.startMal
    a3fl = args.a3fl

    print(args)
    headerFile = headerFile + "/"
    bDoorRefCount = percentages.count(0.0)
    if alpha == 0:
        alpha = None
    # Load Data
    trainLoader, testLoader, malTrainloader = DataAug.getLoaders(numClients, numMal,dataset=dataset,
                                                                 attack_type=attack_type, backdoor=backdoor,alpha=alpha)

    # Take Control of All Malicious Clients
    if numMal > 0:
        malDataset = ConcatDataset(loader.dataset for loader in trainLoader[:numMal])
        malLoader = DataLoader(malDataset, batch_size=64, shuffle=False)
    else:
        malLoader = None

    # Get the backdoored samples available to co-ordinated malicious clients
    _, _, trainloader = DataAug.getLoaders(numClients, numMal,dataset=dataset,attack_type=attack_type,backdoor=backdoor)

    #Run Federated Learning
    g, gAccs, gLosses, gASR, accs, losses, selected, gpreds, cpreds, alphas = FedUtils.trainFedModel(trainLoader, testLoader, malLoader,
                                                                   numClients, trainloader, trainingRounds, epochs,
                                                                   percentages,device,numMal,file=headerFile,
                                                                   verbose=verbose, model=net(),lr=lr,
                                                                   dataset=dataset,bDoorRefCount=bDoorRefCount,
                                                                   scheme=scheme, param=param,adaptive=adaptive,
                                                                   r=r,adaptiveInterval=ai, attack_type=attack_type,
                                                                   asr=asr,backdoor=backdoor,lossFunc=lossFunc,
                                                                                                     startMal=startMal,
                                                                                                     a3fl=a3fl)

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
        clean(headerFile + "FederatedModels")
        clean(headerFile + "ReferenceModels")
        clean(headerFile + "trainloader")
