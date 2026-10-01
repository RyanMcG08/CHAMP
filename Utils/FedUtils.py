import os

import torch
import copy
from sklearn.svm import SVC
import pickle
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,ConcatDataset
from Utils import Training, DataAug
from Utils.Models import *
from itertools import cycle
import numpy as np
import random
def trainFedModel(trainLoader, testLoader, malLoader, numClients,backdooredLoader,trainingRounds,epochs,
                  percentages, device, numMal, file = "", verbose = True,retrainPoint=1,model=BatchNormModel(),
                  C=1,kernel="poly",tol=1e-3,lr = 0.1,dataset=MNIST,bDoorRefCount = 1,
                  scheme = 0, param=0, adaptive=False, r=10, adaptiveInterval = 1,attack_type=0,delta=1,
                  backdoor = DataAug.letter_R, lossFunc=Training.euclidean_dist,startMal = 0,
                  selection = "fixed",save = True,bd_percent = 1, batch_size = 64):
    """
    Runner that runs the federated learning system
    :param trainLoader: Set of trainloaders
    :param testLoader: Set of testloaders
    :param malLoader: Malicious client(s) loader
    :param numClients: Number of clients
    :param backdooredLoader: Poisoned samples loader
    :param trainingRounds: Number of global training rounds
    :param epochs: Number of local training epochs
    :param percentages: Set of values for the poisoning of BSCI inference model reference models
    :param device: Device being run from
    :param numMal: Number of malious clients
    :param file: Output file
    :param verbose: Verbose toggle
    :param retrainPoint: How many epochs to retrain out BSCI model
    :param model: CNN architecture
    :param C: SVC parameter C
    :param kernel: SVC Kernel
    :param tol: SVC parameter tolerance
    :param lr: Learning rate for local training
    :param dataset: Dataset
    :param bDoorRefCount: How many reference models do NOT contain a backdoor
    :param scheme: RA scheme
    :param param: Parameter used for RA scheme if relevant
    :param adaptive: using BSCI to alter loss toggle
    :param r: what training round to begin implementing our adaptive poisoning
    :param adaptiveInterval: What rounds to recalculate alpha
    :param attack_type: Backdoor or label-flipping, targeted or untargeted
    :param delta: Scalar for scaling poisonous loss function
    :param asr: Using ASR as proximity metric toggle
    :param backdoor: What backdoor type
    :param alpha: Dirichlet distribution parameter
    :param lossFunc: What loss function champ uses
    :param selection: How to organise client selection
    :param save: Save output files or not
    :param bd_percent: What percentage of samples to backdoor
    :param batch_size: Batch size for training
    :return: the FL results
    """
    datasets = [loader.dataset for loader in testLoader]
    combined_dataset = ConcatDataset(datasets)

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=64,
        shuffle=False
    )
    selected = []
    nets = [copy.deepcopy(Training.createModel(model, device)) for _ in range(numClients)]

    accs = [[]*numClients for _ in range(numClients)]
    losses = [[]*numClients for _ in range(numClients)]
    gAccs = []
    gLosses = []
    gASRs = []
    gpreds = []
    cpreds = []
    alphas = []
    if selection == "fixed" and numClients > 10:
        full_setup = get_fixed(trainingRounds, numClients,numMal,10,startMal)
    for round in range(trainingRounds):
        if verbose:
            print(f"Round {round + 1}")
        if round > 0:
            global_model = Training.createModel(model, device, file + "FederatedModels/Global"+str(round-1))
        else:
            global_model = Training.createModel(model, device)
        for i in range(numClients):
            nets[i].load_state_dict(global_model.state_dict())
        alpha = -1
        if numClients <= 99:
            Clients = list(range(10))
            for i in range(numClients):
                if verbose:
                    print(f"Training Local Model {i+1}")
                if i < numMal and round > 0 and adaptive is True and (round+1) % adaptiveInterval == 0:
                    alpha = getAvMI(gpreds,r)

                    if verbose:
                        print(f"Alpha {alpha}")
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                   file + "FederatedModels/Model" + str(i) + "_" + str(round),
                                                   False, verbose=verbose, lr=lr, alpha=alpha,round=round,model=model,
                                                    delta=delta,lossFunc=lossFunc,save=save)

                else:
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                    file + "FederatedModels/Model" + str(i) + "_" + str(round),
                                                    False,verbose=verbose,lr=lr,save=save)
                losses[i].append(loss)
                accs[i].append(acc)
            if alpha == -1 and (round+1) % adaptiveInterval == 0 and round > 0:
                alpha = getAvMI(gpreds, r)
        else:
            if round < startMal:
                Clients = random.sample(range(numMal,100), 10)
            elif selection == "random":
                Clients = random.sample(range(100), 10)
            elif selection == "fixed":
                Clients = full_setup[round]
            
            for i in Clients:
                if verbose:
                    print(f"Training Local Model {i+1}")
                if i < numMal and round > 0 and adaptive is True and (round+1) % adaptiveInterval == 0:
                    alpha = getAvMI(gpreds, r)
                    if verbose:
                        print(f"Alpha {alpha}")
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                   file + "FederatedModels/Model" + str(i) + "_" + str(round),
                                                   False, verbose=verbose, lr=lr, alpha=alpha,round=round,model=model,
                                                    delta=delta,lossFunc=lossFunc,save=save)
                else:
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                    file + "FederatedModels/Model" + str(i) + "_" + str(round),
                                                    False,verbose=verbose,lr=lr,save=save)
                losses[i].append(loss)
                accs[i].append(acc)
            if alpha == -1 and (round+1) % adaptiveInterval == 0 and round > 1:
                alpha = getAvMI(gpreds, r)
        alphas.append(alpha)
        nets_ = []
        if Clients != []:
            for i in range(len(Clients)):
                nets_.append(nets[Clients[i]])
        try:
            fed, selected_ = getAgg(nets_,scheme,trainLoader,param)
            if (isinstance(selected_, int) == True):
                if Clients[selected_] < numMal:
                    selected.append(1)
                else:
                    selected.append(0)
            elif (isinstance(selected_, float) == True):
                selected.append(selected_)
            else:
                toggle = 0
                for i in selected_:
                    if Clients[i] < numMal:
                        toggle += selected_[i]
                selected.append(toggle)
        except:
            fed = getAgg(nets, scheme, trainLoader, param)
        gLoss, gAcc = Training.testModel(fed, combined_loader, "Federated Model on test set",verbose=verbose)
        if malLoader != None and (attack_type == 1 or attack_type == 3):
            _, gASR = Training.testModel(fed, backdooredLoader, "Federated Model on all backdoored data in malicious clients",verbose=verbose, asr=True)
            gASRs.append(gASR)
        elif malLoader != None and (attack_type == 0 or attack_type == 2):
            _, gASR = Training.testModel(fed, backdooredLoader, "Federated Model on all backdoored data in malicious clients",verbose=verbose, asr=False)
            gASRs.append(gASR)
        gLosses.append(gLoss)
        gAccs.append(gAcc)

        if save: torch.save(fed.state_dict(), file + "FederatedModels/Global"+str(round))
        else:
            if round == 0 and not os.path.exists(file+ "FederatedModels/"): os.makedirs(file + "FederatedModels/")
            torch.save(fed.state_dict(), file + "FederatedModels/Global" + str(round))
            if round != 0: os.remove(file + "FederatedModels/Global"+str(round-1))

        if round >= startMal-1 and adaptive == 1:
            if round % retrainPoint == 0:
                if verbose:
                    print("Training Reference Models")
                # Train Reference Models
                if malLoader != None:
                    refNets, refLosses, refAccs = Training.trainRefModels(malLoader, percentages,
                                                                          epochs=epochs, device=device, file=file,
                                                                          verbose=verbose,
                                                                          startingPoint=file + "FederatedModels/Global" + str(
                                                                              round), round=round, model=model,lr=lr,
                                                                          dataset=dataset,backdoor=backdoor,save=save)
                else:
                    malDataset = ConcatDataset(loader.dataset for loader in trainLoader[:int(numClients/2)])
                    malLoader = DataLoader(malDataset, batch_size=64, shuffle=False)
                    refNets, refLosses, refAccs = Training.trainRefModels(malLoader, percentages,
                                                                          epochs=epochs, device=device, file=file,
                                                                          verbose=verbose,
                                                                          startingPoint=file + "FederatedModels/Global" + str(
                                                                              round), round=round, model=model,lr=lr,
                                                                          dataset=dataset,backdoor=backdoor,save=save)
                    malLoader = None
                    _, _, backdooredLoader = DataAug.getLoaders(numClients, int(numClients/2), dataset=dataset,
                                                                attack_type=attack_type, bd_percent=bd_percent, bs=batch_size)


                refFVS, refLabels = Training.getFVS(refNets, backdooredLoader, training=True,bDoorRefCount=bDoorRefCount)

                if verbose:
                    print("Training Classifier")
                classifier = SVC(kernel=kernel, probability=True, C=C,tol=tol)
                classifier.fit(refFVS, refLabels)
                fed_preds, fed_FVS = Training.getPrediction(classifier,[fed],backdooredLoader,["Global Model"],verbose=verbose)
                pred = (np.sum(fed_preds) / len(fed_preds[0])) * 100
                gpreds.append(pred)
                if numClients <= 99:
                    client_preds, client_FVS = Training.getPrediction(classifier,
                                                            nets_,
                                                            backdooredLoader,
                                                            None,verbose=verbose)
                    cpreds_ = []
                    for client in Clients:
                        cpreds_.append((np.sum(client_preds[client]) / len(client_preds[client])) * 100)
                    cpreds.append(cpreds_)

    if save: pickle.dump(backdooredLoader, open(file + "trainloader", "wb"))

    return fed, gAccs, gLosses, gASRs, accs,losses, selected, gpreds, cpreds, alphas

def get_fixed(trainingRounds, numClients,numMal,clients_per_round,startMal):
    """
    get organisation of all clients selection
    :param trainingRounds: Number of training rounds
    :param numClients: Number of Clients
    :param numMal: Number of Malicious Clients
    :param clients_per_round: Number of Malicious Clients per aggregation round
    :param startMal: What round malicious clients start attacking
    :return: What clients to select in each round
    """
    random.seed(42)
    malicious_clients = list(range(numMal))
    honest_clients = list(range(numMal, numClients))
    malicious_cycle = cycle(malicious_clients)


    malicious_counter = 0
    round_schedule = []

    for round in range(trainingRounds):
        if round < startMal:
            clients = random.sample(honest_clients, clients_per_round)
        else:
            malicious_counter += clients_per_round * (numMal / numClients)
            malicious_counter -=  int(malicious_counter)

            selected_malicious = [next(malicious_cycle) for _ in range(int(malicious_counter))]
            selected_honest = random.sample(honest_clients, clients_per_round - int(malicious_counter))

            clients = selected_malicious + selected_honest
            random.shuffle(clients)

        round_schedule.append(clients)
    return round_schedule
def getAgg(nets, scheme, trainloader,param):
    """
    get and run RA scheme
    :param nets: Uploaded models to the server
    :param scheme: RA scheme selection
    :param trainloader: set of trainloaders
    :param param: Parameters for RA scheme if applicable
    :param numMal: Number of malicious clients
    :return: New Global Model
    """
    if scheme == 0:
        return weightedAvg(nets, trainloader)
    elif scheme == 1:
        return median(nets)
    elif scheme == 2:
        return fta(nets,param)
    elif scheme == 3:
        return krum(nets,param)
    elif scheme == 4:
        return bulyan(nets,param)
    elif scheme == 5:
        return rfa(nets)
    elif scheme == 6:
        return dai(nets)
    else:
        assert "No Valid Aggregation Scheme Selected"
def weightedAvg(nets, trainLoaders):
    """
    Aggregate the models using weighted averaging (FedAvg).
    :param nets: list of models from clients
    :param trainLoaders: list of train loaders to get dataset sizes
    :return: aggregated model
    """
    fed = copy.deepcopy(nets[0])
    num_samples = [len(loader.dataset) for loader in trainLoaders]
    total_samples = sum(num_samples)

    with torch.no_grad():
        for param in fed.parameters():
            param.zero_()

        for model, samples in zip(nets, num_samples):
            for avg_param, model_param in zip(fed.parameters(), model.parameters()):
                avg_param.add_((samples / total_samples) * model_param)

    return fed

def Avg(nets):
    """
    Aggregate the models using FedAvg with no weight on local training set size.
    :param nets: list of models from clients
    :return: aggregated model
    """
    fed = copy.deepcopy(nets[0])

    with torch.no_grad():
        for param in fed.parameters():
            param.zero_()

        for model in nets:
            for avg_param, model_param in zip(fed.parameters(), model.parameters()):
                avg_param.add_((1/len(nets)) * model_param)

    return fed

def median(nets):
    """
    Aggregate the models using median.
    :param nets: list of models from clients
    :return: aggregated model
    """
    fed = copy.deepcopy(nets[0])

    with torch.no_grad():
        for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in nets]):
            stacked = torch.stack(params)
            median_param = torch.median(stacked, dim=0).values
            fed_param.data.copy_(median_param)

    return fed

def fta(nets, beta=0.1):
    """
    Coordinate-wise trimmed mean aggregation.
    :param nets: list of client models
    :param beta: fraction to trim from each side
    :return: aggregated model
    """
    fed = copy.deepcopy(nets[0])
    num_clients = len(nets)
    k = int(beta * num_clients)

    with torch.no_grad():
        for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in nets]):
            stacked = torch.stack(params)
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed_vals = sorted_vals[k:num_clients - k] if num_clients - 2*k > 0 else sorted_vals
            mean_param = trimmed_vals.mean(dim=0)
            fed_param.data.copy_(mean_param)

    return fed


def krum(nets, m=1):
    """
    Krum (or Multi-Krum) aggregation.
    :param nets: list of client models
    :param m: number of models to average in Multi-Krum
    :param f: number of malicious clients allowed by the aggregator
    :return: aggregated model, selected idx(s), ranking list (ranks_by_client[i] = rank of client i)
    """
    f = 1
    num_clients = len(nets)
    flat_params = []

    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        flat_params.append(vec)

    distances = torch.zeros(num_clients, num_clients)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.norm(flat_params[i] - flat_params[j]) ** 2
            distances[i, j] = distances[j, i] = dist

    scores = []
    for i in range(num_clients):
        dists = distances[i].clone()
        nearest = torch.topk(dists, k=int(num_clients - f - 1), largest=False).values
        scores.append(torch.sum(nearest).item())

    scores_tensor = torch.tensor(scores)
    ranked_idxs = torch.argsort(scores_tensor).tolist()
    ranks_by_client = [0] * num_clients
    for rank, cid in enumerate(ranked_idxs):
        ranks_by_client[cid] = rank

    if m > 1:
        selected_idxs = torch.topk(scores_tensor, k=int(m), largest=False).indices
        selected_models = [nets[i] for i in selected_idxs]

        fed = copy.deepcopy(nets[0])
        with torch.no_grad():
            for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in selected_models]):
                stacked = torch.stack(params)
                mean_param = stacked.mean(dim=0)
                fed_param.data.copy_(mean_param)

        return fed, ranks_by_client

    else:
        best_idx = torch.argmin(scores_tensor).item()
        fed = copy.deepcopy(nets[best_idx])
        return fed, ranks_by_client

def bulyan(nets, f=1):
    """
    Bulyan aggregation: combines Multi-Krum and trimmed mean.

    :param nets: list of client models
    :param f: number of Byzantine clients to tolerate
    :return: aggregated model, selected models info, ranking list (ranks_by_client[i] = rank of client i)
    """
    num_clients = len(nets)
    assert 2 * f + 3 <= num_clients, "Not enough clients for Bulyan (requires at least 2f + 3)"

    flat_params = []
    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        flat_params.append(vec)

    distances = torch.zeros(num_clients, num_clients)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.norm(flat_params[i] - flat_params[j]) ** 2
            distances[i, j] = distances[j, i] = dist

    scores = []
    for i in range(num_clients):
        dists = distances[i].clone()
        nearest = torch.topk(dists, k=int(num_clients - f - 2), largest=False).values
        scores.append(torch.sum(nearest).item())

    scores_tensor = torch.tensor(scores)
    ranked_idxs = torch.argsort(scores_tensor).tolist()

    ranks_by_client = [0] * num_clients
    for rank, cid in enumerate(ranked_idxs):
        ranks_by_client[cid] = rank

    num_selected = num_clients - (2 * f)
    selected_idxs = torch.topk(scores_tensor, k=int(num_selected), largest=False).indices
    selected_models = [nets[i] for i in selected_idxs]

    fed = copy.deepcopy(nets[0])
    with torch.no_grad():
        for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in selected_models]):
            stacked = torch.stack(params)
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed_vals = sorted_vals[int(f):int(num_selected - f)] if int(num_selected) - 2 * f > 0 else sorted_vals
            mean_param = trimmed_vals.mean(dim=0)
            fed_param.data.copy_(mean_param)

    return fed, ranks_by_client

def rfa(nets, max_iter=50, tol=1e-6, eps=1e-6):
    """
    Robust Federated Aggregation (RFA)
    :param nets: list of client models
    :param max_iter: max number of iterations for the geometric median solver
    :param tol: relative convergence tolerance
    :param eps: stability constant for distance smoothing
    :return: aggregated model
    """
    flat_params = []
    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        flat_params.append(vec)

    stacked_params = torch.stack(flat_params)
    median = stacked_params.mean(dim=0)
    for _ in range(max_iter):
        diffs = stacked_params - median.unsqueeze(0)
        distances = torch.norm(diffs, dim=1)

        distances = torch.clamp(distances, min=eps)

        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_median = (weights.unsqueeze(1) * stacked_params).sum(dim=0)
        if torch.norm(new_median - median) / (torch.norm(median) + eps) < tol:
            median = new_median
            break

        median = new_median
    fed = copy.deepcopy(nets[0])
    with torch.no_grad():
        pointer = 0
        for p in fed.parameters():
            numel = p.numel()
            p.data.copy_(median[pointer:pointer + numel].view_as(p))
            pointer += numel

    return fed, weights[0].item()

def dai(nets, numMal=1):
    """
    Direction Alignment Inspection (DAI)

    :param nets: list of client models
    :param threshold_quantile: unused if numMal is set meaningfully; kept for backward compat
    :param numMal: number of clients to treat as malicious and filter out
    :return: aggregated model, indices of benign clients, ranking list (ranks_by_client[i] = rank of client i, 0 = most aligned)
    """
    flat_params = []
    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        flat_params.append(vec)

    stacked_params = torch.stack(flat_params)
    directions = stacked_params - stacked_params.mean(dim=0, keepdim=True)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)

    similarity_matrix = torch.matmul(directions, directions.T)
    num_clients = similarity_matrix.shape[0]
    alignment_scores = (similarity_matrix.sum(dim=1) - 1) / (num_clients - 1)

    ranked_idxs = torch.argsort(alignment_scores, descending=True).tolist()

    ranks_by_client = [0] * num_clients
    for rank, cid in enumerate(ranked_idxs):
        ranks_by_client[cid] = rank


    k = max(num_clients - numMal, 1)
    benign_indices = torch.topk(alignment_scores, k=k, largest=True).indices

    filtered_nets = [nets[i] for i in benign_indices.tolist()]
    fed = Avg(filtered_nets)

    return fed, ranks_by_client


def getAvMI(preds, r):
    """
    Compute the average alpha value for training
    :param preds: predictions on the global model during training
    :param r: how many rounds to consider in the calculation of alpha
    :return: Alpha
    """
    if len(preds) < r:
        return 0

    running_average = 0
    for i in range(1,r+1):
        running_average += preds[-i]
    running_average /= (r*100)
    return 1 - running_average

def getAvASR(asrs, r):
    """
    Compute the average alpha value for training when using ASR not the BSCI model
    :param asrs: Global model ASR during training
    :param r: how many rounds to consider in the calculation of alpha
    :return: Alpha
    """
    if len(asrs) < r:
        return 0

    running_average = 0
    for i in range(1,r+1):
        running_average += asrs[-i]
    running_average /= (r*100)
    return 1 - running_average