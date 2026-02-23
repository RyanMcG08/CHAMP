import os

from AlexNet import *
import torch
import copy
from sklearn.svm import SVC
import pickle
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader,ConcatDataset
from Utils import Training, DataAug, FedUtils, PlottingUtils, A3fl
from itertools import cycle
import numpy as np
import random

def trainFedModel(trainLoader, testLoader, malLoader, numClients,backdooredLoader,trainingRounds,epochs,
                  percentages, device, numMal, file = "", verbose = True,retrainPoint=1,model=alexnet(),
                  C=1,kernel="poly",tol=1e-3,lr = 0.1,dataset=MNIST,bDoorRefCount = 1,
                  scheme = 0, param=0, adaptive=False, r=10, adaptiveInterval = 1,attack_type=0,delta=1,asr=False,
                  backdoor = DataAug.letter_R, lossFunc=Training.euclidean_dist,startMal = 0,a3fl = False,
                  selection = "fixed",save = True):
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
    :param attack_type: Backdoor or label-flipping, targeted r untargeted
    :param delta: Scalar for scaling poisonous loss function
    :param asr: Using ASR as proximity metric toggle
    :param backdoor:
    :param alpha:
    :param lossFunc:
    :return:
    """
    if a3fl:
        # Initialize attacker
        adv_epochs = 5
        trigger_lr = 0.01
        trigger_outter_epochs = 10
        dm_adv_K = 1
        dm_adv_model_count = 1
        noise_loss_lambda = 0.01
        bkd_ratio = 1
        if backdoor == DataAug.onebyone: ts = 1
        elif backdoor == DataAug.threebythree: ts = 3
        else: ts = 5
        if dataset == MNIST or dataset == FashionMNIST: [channel,im_size] = 1,28
        else: [channel,im_size] = 3,32
        attacker = A3fl.Attacker(ts, adv_epochs, 1, trigger_lr, trigger_outter_epochs,
                            dm_adv_K, dm_adv_model_count, noise_loss_lambda, bkd_ratio,channel,im_size)
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
    if selection == "fixed":
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
                if i < numMal and round > 0 and adaptive is True and (round+1) % adaptiveInterval == 0 and a3fl == False:
                    alpha = getAvMI(gpreds,r)
                    if asr:
                        alpha = getAvASR(gASRs,r)
                    if verbose:
                        print(f"Alpha {alpha}")
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                   file + "FederatedModels/Model" + str(i) + str(round),
                                                   False, verbose=verbose, lr=lr, alpha=alpha,round=round,model=model,
                                                    delta=delta,lossFunc=lossFunc,save=save)
                elif i < numMal and round > 0 and (round+1) % adaptiveInterval == 0 and a3fl == True:
                    loss, acc, mask, trigger = A3fl.RunAttack(nets[i], trainLoader[i], epochs,global_model,attacker,device, verbose=verbose, lr=lr,
                                              round=round)
                    backdoor = [mask,trigger]
                else:
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                    file + "FederatedModels/Model" + str(i) + str(round),
                                                    False,verbose=verbose,lr=lr,save=save)
                losses[i].append(loss)
                accs[i].append(acc)
            if alpha == -1 and (round+1) % adaptiveInterval == 0:
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
                    alpha = getAvMI(gpreds,r)
                    if asr:
                        alpha = getAvASR(gASRs,r)
                    if verbose:
                        print(f"Alpha {alpha}")
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                   file + "FederatedModels/Model" + str(i) + str(round),
                                                   False, verbose=verbose, lr=lr, alpha=alpha,round=round,model=model,
                                                    delta=delta,lossFunc=lossFunc,save=save)
                elif i < numMal and round > 0 and (round+1) % adaptiveInterval == 0 and a3fl == True:
                    loss, acc, mask, trigger = A3fl.RunAttack(nets[i], trainLoader[i], epochs,global_model,attacker,device, verbose=verbose, lr=lr,
                                              round=round)
                    backdoor = [mask, trigger]
                else:
                    loss, acc = Training.trainModel(nets[i], epochs, trainLoader[i], testLoader[i], device,
                                                    file + "FederatedModels/Model" + str(i) + str(round),
                                                    False,verbose=verbose,lr=lr,save=save)
                losses[i].append(loss)
                accs[i].append(acc)
            if alpha == -1 and (round+1) % adaptiveInterval == 0:
                alpha = getAvMI(gpreds, r)
        alphas.append(alpha)
        nets_ = []
        if Clients != []:
            for i in range(len(Clients)):
                nets_.append(nets[Clients[i]])
        try:
            fed, selected_ = getAgg(nets_,scheme,trainLoader,param, global_model,numMal,round,file)
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
                        toggle += 1
                selected.append(toggle)
        except:
            fed = getAgg(nets, scheme, trainLoader, param, global_model,numMal,round,file)
        gLoss, gAcc = Training.testModel(fed, testLoader[0], "Federated Model on client 0 test data",verbose=verbose)
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
                                                                attack_type=attack_type)

                # Get feature vectors from reference models across backdoored samples and reference models across backdoored samples
                refFVS, refLabels = Training.getFVS(refNets, backdooredLoader, training=True,bDoorRefCount=bDoorRefCount)
                # Train SVC classifier on reference models Feature Vectors
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
    random.seed(42)
    malicious_clients = list(range(numMal))
    honest_clients = list(range(numMal, numClients))

    # Cycle through malicious clients to rotate them fairly
    malicious_cycle = cycle(malicious_clients)

    # Counter for fractional malicious slots
    malicious_counter = 0

    # For storing selected clients per round
    round_schedule = []

    for round in range(trainingRounds):
        if round < startMal:
            # before malicious clients start showing up
            clients = random.sample(honest_clients, clients_per_round)
        else:
            # Calculate expected number of malicious clients this round
            expected_malicious = clients_per_round * (numMal / numClients)
            malicious_counter += expected_malicious

            # Number of malicious clients to actually pick this round
            num_malicious_to_pick = int(malicious_counter)
            malicious_counter -= num_malicious_to_pick  # subtract what we picked

            # Pick malicious clients fairly using the cycle
            selected_malicious = [next(malicious_cycle) for _ in range(num_malicious_to_pick)]

            # Pick remaining honest clients
            num_honest_to_pick = clients_per_round - num_malicious_to_pick
            selected_honest = random.sample(honest_clients, num_honest_to_pick)

            clients = selected_malicious + selected_honest
            random.shuffle(clients)  # optional: mix malicious and honest

        round_schedule.append(clients)
    return round_schedule
def getAgg(nets, scheme, trainloader, param,g,numMal,round,file):
    """
    get and run RA scheme
    :param nets: Uploaded models to the server
    :param scheme: RA scheme selection
    :param trainloader: set of trainloaders
    :param param: Parameters for RA scheme if applicable
    :param g: Previous round global model
    :param numMal: Number of malicious clients
    :param round: Training Round
    :return: New Global Model
    """
    if scheme == 0:
        return weightedAvg(nets, trainloader)
    elif scheme == 1:
        return median(nets)
    elif scheme == 2:
        return fta(nets,param)
    elif scheme == 3:
        return krum(nets,param,f=numMal)
    elif scheme == 4:
        return bulyan(nets,param,numMal)
    elif scheme == 5:
        return rfa(nets)
    elif scheme == 6:
        return dai(nets,param,numMal)
    elif scheme == 7:
        return aggregate_with_rlr(nets,g)
    elif scheme == 8:
        return aggregate_with_foolsgold(nets,g)
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


def krum(nets, m=1, f=1):
    """
    Krum (or Multi-Krum) aggregation.
    :param nets: list of client models
    :param m: number of models to average in Multi-Krum
    :param f: number of malicious clients allowed by the aggregator
    :return: aggregated model
    """
    numMal = f
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

    # Compute Krum scores
    scores = []
    for i in range(num_clients):
        dists = distances[i].clone()
        nearest = torch.topk(dists, k=int(num_clients - f - 1), largest=False).values
        scores.append(torch.sum(nearest).item())

    # Select model(s) with lowest scores
    if m > 1:
        selected_idxs = torch.topk(torch.tensor(scores), k=int(m), largest=False).indices
        selected_models = [nets[i] for i in selected_idxs]

        fed = copy.deepcopy(nets[0])
        with torch.no_grad():
            for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in selected_models]):
                stacked = torch.stack(params)
                mean_param = stacked.mean(dim=0)
                fed_param.data.copy_(mean_param)

        return fed, selected_idxs

    else:
        best_idx = torch.argmin(torch.tensor(scores)).item()
        fed = copy.deepcopy(nets[best_idx])
        return fed, best_idx

def bulyan(nets, f=1,numMal=1):
    """
    Bulyan aggregation: combines Multi-Krum and trimmed mean.

    :param nets: list of client models
    :param f: number of Byzantine clients to tolerate
    :param numMal: Number of malicious clients in the system
    :return: aggregated model
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

    num_selected = num_clients - (2 * f)
    selected_idxs = torch.topk(torch.tensor(scores), k=int(num_selected), largest=False).indices
    selected_models = [nets[i] for i in selected_idxs]

    fed = copy.deepcopy(nets[0])
    with torch.no_grad():
        for fed_param, *params in zip(fed.parameters(), *[net.parameters() for net in selected_models]):
            stacked = torch.stack(params)
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed_vals = sorted_vals[int(f):int(num_selected - f)] if int(num_selected) - 2 * f > 0 else sorted_vals
            mean_param = trimmed_vals.mean(dim=0)
            fed_param.data.copy_(mean_param)

    if any(i < f for i in selected_idxs):
        models = [i.item() for i in selected_idxs if i.item() < numMal]
        return fed, models
    else:
        return fed, 0

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

def dai(nets, threshold_quantile=0.1,numMal=1):
    """
    Direction Alignment Inspection (DAI)

    :param nets: list of client models
    :param threshold_quantile: quantile threshold to filter clients with lowest alignment scores
    :param numMal: Number of malicious clients in system
    :return: list of filtered client models
    """
    flat_params = []
    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        flat_params.append(vec)

    stacked_params = torch.stack(flat_params)
    directions = stacked_params - stacked_params.mean(dim=0, keepdim=True)
    directions = directions / directions.norm(dim=1, keepdim=True)

    similarity_matrix = torch.matmul(directions, directions.T)
    num_clients = similarity_matrix.shape[0]
    alignment_scores = (similarity_matrix.sum(dim=1) - 1) / (num_clients - 1)

    threshold = torch.quantile(alignment_scores, threshold_quantile)
    benign_indices = (alignment_scores >= threshold).nonzero(as_tuple=True)[0]
    filtered_nets = [nets[i] for i in benign_indices.tolist()]

    fed = Avg(filtered_nets)

    if any(i < numMal for i in benign_indices):
        models = [benign_indices[i].item() for i in benign_indices if i < numMal]
        return fed, models
    else:
        return fed, 0


def robust_learning_rates_from_updates(updates, c=1.0, eps=1e-12):
    norms = torch.norm(updates, dim=1)
    scales = c / (c + norms + eps)
    return scales, norms

def aggregate_with_rlr(nets, reference_net, c=1.0, theta=1.0, flip_when_ambiguous=True):
    """
    RLR
    :param nets: list of client models
    :param reference_net: Previous rounds global model for reference
    :param c: RLR hyperparameter controlling scaling
    :param theta: threshold for sign-of-signs decision
    :param flip_when_ambiguous: Multiplies LR by -1 when sum_signs < theta if True, if False these co-ords are zeroed
    :return: Aggregated model and the scalar value at each round as a measure of trust in malicious client
    """
    ref_vec = torch.cat([p.data.view(-1) for p in reference_net.parameters()])
    updates = []
    for net in nets:
        client_vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        updates.append(client_vec - ref_vec)
    stacked_updates = torch.stack(updates)
    scales, norms = robust_learning_rates_from_updates(stacked_updates, c=c)

    signs = torch.sign(stacked_updates)
    weighted_sign_sums = (scales.unsqueeze(1) * signs).sum(dim=0)

    # Determine base direction by sign-of-signs
    final_direction = torch.sign(weighted_sign_sums)

    # Handle ambiguous coordinates: abs(sum) < theta
    ambiguous_mask = weighted_sign_sums.abs() < theta

    if ambiguous_mask.any():
        if flip_when_ambiguous:
            final_direction[ambiguous_mask] = -final_direction[ambiguous_mask]
            zero_mask = final_direction == 0
            if zero_mask.any():
                final_direction[zero_mask] = -1.0
        else:
            final_direction[ambiguous_mask] = 0.0

    weighted_abs = (scales.unsqueeze(1) * stacked_updates.abs()).sum(dim=0)
    denom = scales.sum().clamp_min(1e-12)
    avg_weighted_abs = weighted_abs / denom

    final_update = final_direction * avg_weighted_abs
    new_vec = ref_vec + final_update
    aggregated = copy.deepcopy(reference_net)
    with torch.no_grad():
        pointer = 0
        for p in aggregated.parameters():
            numel = p.numel()
            p.data.copy_(new_vec[pointer:pointer + numel].view_as(p))
            pointer += numel

    return aggregated, scales[0].item()


def foolsgold_weights(nets):
    """
    :param nets: Uploaded nets to the central server
    :return: The weights for the foolsgold Ra scheme (trust in each client)
    """
    n = len(nets)
    updates = []
    for net in nets:
        vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        updates.append(vec)
    updates = torch.stack(updates)

    updates_norm = updates / (updates.norm(dim=1, keepdim=True) + 1e-10)

    cs = torch.mm(updates_norm, updates_norm.t())
    cs.fill_diagonal_(0)
    maxcs, _ = cs.max(dim=1)
    for i in range(n):
        for j in range(n):
            if maxcs[i] < maxcs[j]:
                cs[i, j] *= maxcs[i] / maxcs[j]

    weights = 1 - cs.max(dim=1)[0]
    weights = torch.clamp(weights, 0, 1)

    eps = 1e-10
    weights = torch.log(weights / (weights + eps) + eps)
    weights = torch.sigmoid(weights * 10)

    if weights.max() > 0:
        weights = weights / weights.max()

    return weights


def aggregate_with_foolsgold(nets, reference_net):
    """
    Aggregate client models using FoolsGold scaling.

    :param nets: Client models
    :param reference_net: Previous round global model
    :return: New global model
    """
    ref_vec = torch.cat([p.data.view(-1) for p in reference_net.parameters()])

    weights = foolsgold_weights(nets)

    updates = []
    for net in nets:
        client_vec = torch.cat([p.data.view(-1) for p in net.parameters()])
        update = client_vec - ref_vec
        updates.append(update)
    stacked_updates = torch.stack(updates)

    weighted_update = (weights.unsqueeze(1) * stacked_updates).sum(dim=0) / weights.sum()
    new_vec = ref_vec + weighted_update

    aggregated = copy.deepcopy(reference_net)
    with torch.no_grad():
        pointer = 0
        for p in aggregated.parameters():
            numel = p.numel()
            p.data.copy_(new_vec[pointer:pointer + numel].view_as(p))
            pointer += numel

    return aggregated, weights[0].item()

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