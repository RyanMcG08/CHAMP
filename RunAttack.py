import numpy as np
import torch
import os
from Utils import Training, DataAug, FedUtils, PlottingUtils
from torch.utils.data import DataLoader,ConcatDataset
from AlexNet import *
import shutil
from torchvision.datasets import CIFAR10, FashionMNIST, CIFAR100, MNIST
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import argparse


class TinyImageNet(Dataset):
    def __init__(self, root=None, train=True, download=True, transform=None):
        dataset_dict = load_dataset("zh-plus/tiny-imagenet")

        if train:
            self.data = dataset_dict["train"]
        else:
            self.data = dataset_dict["valid"]

        self.transform = transform

        # ✅ ADD THIS (torchvision compatibility)
        self.targets = [item["label"] for item in self.data]

        # Optional but nice (some codebases expect this)
        try:
            self.classes = self.data.features["label"].names
        except:
            self.classes = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

def clean(file):
    try:
        shutil.rmtree(file)
    except:
        os.remove(file)

def getModel(model_name):
    if model_name == 'alexnet':
        return alexNetCifar
    elif model_name == 'fashionMNISTCNN':
        return FashionMNIST_CNN
    elif model_name == 'fashionMNIST':
        return ResNet18_fashionMNIST
    elif model_name == 'fashionMNISTBN':
        return ResNet18_fashionMNISTBN
    elif model_name == 'cifar10':
        return ResNet18_cifar10
    elif model_name == 'cifar10BN':
        return ResNet18_cifar10BN
    elif model_name == 'cifar100':
        return ResNet18_cifar100
    elif model_name == 'cifar100BN':
        return ResNet18_cifar100BN
    elif model_name == 'imagenet':
        return ResNet18_tinyImageNet
    elif model_name == 'imagenetBN':
        return ResNet18_tinyImageNetBN

    elif model_name == 'resnetNoBN':
        return ResNetNoBN
    elif model_name == "alexnetBN":
        return alexNetCifarBN
    elif model_name == "BatchNormOn":
        return BatchNormModel
    elif model_name == "BatchNormOff":
        return NonBatchNormModel
    elif model_name == "alexNetImagenet":
        return alexNetImagenet
    elif model_name == "alexNetImagenetBN":
        return alexNetImagenetBN
    elif model_name == 'VGG16':
        return VGG16
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
    elif dataset_name == "imagenet":
        return TinyImageNet
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
    modelChoices = ["alexnet","alexnetBN","fashionMNISTCNN","resnet","BatchNormOff","BatchNormOn", "cifar10","cifar10BN","cifar100","cifar100BN",'ResNetNoBN',
           'imagenet','imagenetBN','ResNet18_cifar10', 'ResNet18_tinyImageNet', 'fashionMNIST', 'fashionMNISTBN', 'VGG16','GN','LN']
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
    parser.add_argument("--net",type=str, choices=modelChoices, default="fashionMNISTCNN", help="Model name (default: alexnet)")
    parser.add_argument("--dataset", type=str, choices=["mnist","cifar10", "cifar100", "fashionMNIST", "imagenet"], default="fashionMNIST", help="Dataset name (default:Cifar10")
    parser.add_argument("--backdoor", type=str, choices=["one", "three", "five", "LetterR"], default="three", help="Backdoor Type (default: one)")
    parser.add_argument("--alpha", type=float, default=0, help="Parameter alpha for IID level (default: 0)")
    parser.add_argument("--lossFunc", type=int, default=0, help="Loss Function (default: 0)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--startMal", type=int, default=0, help="starting mal behaviour (default: 0)")
    parser.add_argument("--a3fl", type=int, default=0, help="toggle a3fl behaviour (default: 0)")
    parser.add_argument("--selection", type=str, default="fixed", help="aggregator selection (default: fixed)")
    parser.add_argument("--save", type=int, default=1, help="toggle saving models (default: 1)")
    parser.add_argument("--bd_percent", type=float, default=1, help="percentage to backdoor")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lira", type=int, default=0, help="Lira (offline toggle)")


    args = parser.parse_args()
    # Convert 0/1 flags to booleans
    args.verbose = bool(args.verbose)
    args.adaptive = bool(args.adaptive)
    args.asr = bool(args.asr)
    args.cleanTog = bool(args.cleanTog)
    args.a3fl = bool(args.a3fl)
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
    save = args.save
    bd_percent = args.bd_percent
    batch_size = args.batch_size
    lira = args.lira
    print(args)
    headerFile = headerFile + "/"
    bDoorRefCount = percentages.count(0.0)
    # Load Data
    trainLoader, testLoader, malTrainloader = DataAug.getLoaders(numClients, numMal,dataset=dataset,
                                                                 attack_type=attack_type, backdoor=backdoor,alpha=alpha,
                                                                 bd_percent=bd_percent, bs=batch_size)

    # Take Control of All Malicious Clients
    if numMal > 0:
        malDataset = ConcatDataset(loader.dataset for loader in trainLoader[:numMal])
        malLoader = DataLoader(malDataset, batch_size=64, shuffle=False)
    else:
        malLoader = None

    # Get the backdoored samples available to co-ordinated malicious clients
    _, _, trainloader = DataAug.getLoaders(numClients, numMal,dataset=dataset,attack_type=attack_type,backdoor=backdoor,alpha=alpha,
                                           bd_percent=bd_percent, bs=batch_size)
    detector = None
    if lira == 1:
        from Utils.Lira import LiraBackdoorDetector, BackdoorConfig, LiraConfig

        detector = LiraBackdoorDetector(
            model_fn=net,
            backdoor_cfg=BackdoorConfig(poison_frac=0.1, target_label=1),
            lira_cfg=LiraConfig(n_shadow=2, epochs=15), # 8 15
        )
        detector.fit(malLoader,dataset)

        detector.save("checkpoints/lira_detector.pt")

    g, gAccs, gLosses, gASR, accs, losses, selected, gpreds, cpreds, alphas = FedUtils.trainFedModel(trainLoader, testLoader, malLoader,
                                                                   numClients, trainloader, trainingRounds, epochs,
                                                                   percentages,device,numMal,file=headerFile,
                                                                   verbose=verbose, model=net(),lr=lr,
                                                                   dataset=dataset,bDoorRefCount=bDoorRefCount,
                                                                   scheme=scheme, param=param,adaptive=adaptive,
                                                                   r=r,adaptiveInterval=ai, attack_type=attack_type,
                                                                   asr=asr,backdoor=backdoor,lossFunc=lossFunc,
                                                                                                     startMal=startMal,
                                                                                                     a3fl=a3fl,save=save,
                                                                                                     bd_percent = bd_percent,
                                                                                                     batch_size = batch_size,
                                                                                                     detector = detector)

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