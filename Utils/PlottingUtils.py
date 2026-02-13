import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as pe
import numpy as np
import csv
def plotLabelist(allLoader,Title):
    """
    Function to display the labels in each client (used more for non-IID settings)
    :param allLoader: all client loaders
    :param Title: Title of the plot
    :return: displays a plot of all labels
    """
    plt.rcParams.update({'font.size': 18})
    num_clients = len(allLoader)
    num_classes = 10  # CIFAR-10 has 10 classes

    label_distributions = []
    # Step 1: Count labels for each client's dataloader
    for loader in allLoader:
        label_count = [0] * num_classes
        for _, labels in loader:
            for label in labels:
                label_count[label.item()] += 1
        label_distributions.append(label_count)

    # Step 2: Convert to NumPy array for easier plotting
    label_distributions = np.array(label_distributions)

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)

    for i in range(num_clients):
        ax.bar(x + i * 0.08, label_distributions[i], width=0.08, label=f'Client {i}')

    ax.set_xticks(x + 0.36)  # Center the tick labels
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Number of Samples')
    ax.legend()
    plt.tight_layout()
    plt.savefig(Title + ".pdf", dpi=500)
    plt.show()

def TSNE_PLOT(fvs,preds,file="",verbose=True):
    """
    Plots the TSNE of the output feature vectors of the global model at each round
    :param fvs: Feature vectors of the global model
    :param preds: Predicitons of the global model's maliciousness
    :param file: output file
    :param verbose: Verbose toggle
    :return: Tsne plot
    """
    num_models = len(fvs)
    fed_FVS_flat = np.vstack(fvs)
    fed_preds_flat = np.concatenate(preds)

    # Generate client IDs with correct lengths
    model_ids = np.concatenate([np.full(len(fvs[i]), i) for i in range(num_models)])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(fed_FVS_flat)

    # Plot
    plt.figure(figsize=(10, 7))

    for model_id in range(num_models):
        idx = np.where(model_ids == model_id)[0]  # Get correct indices
        client_preds = fed_preds_flat[idx]
        color = 'red' if np.mean(client_preds) > 0.5 else 'blue'
        plt.scatter(
            tsne_results[idx, 0], tsne_results[idx, 1],
            c=color, alpha=0.7,
            edgecolors='k'  # Add edge color for contrast
        )

        mean_tsne = np.mean(tsne_results[idx], axis=0)

        plt.text(
            mean_tsne[0], mean_tsne[1], str(model_id),
            color='black', ha='center', va='center', size=20,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')]  # Add white outline
        )

    plt.scatter([], [], c='red', alpha=0.7, label="Malicious")
    plt.scatter([], [], c='blue', alpha=0.7, label="Benign")
    plt.legend()

    plt.title("t-SNE Visualization of Feature Vectors of Global Model at Different Training Rounds")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.savefig(file + "TSNE.pdf")
    if verbose:
        plt.show()
    plt.close()

def GlobalPlots(accs,asrs,losses,file="",verbose=True):
    """
    Saves global plots to output file
    :param accs: Accuracy of the global model over training rounds
    :param asrs: ASR of the global model over training rounds
    :param losses: Loss of the global model over training rounds
    :param file: Output file
    :param verbose: Verbose toggle
    :return: Saves global plots to output file
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    #ACC
    ax1.set_xlabel("Training Rounds")
    ax1.set_ylabel("Accuracy")
    line1, = ax1.plot(np.arange(1, int(len(accs)+1)),accs, label="Accuracy")
    ax1.tick_params(axis="y")

    #ASR
    ax2 = ax1.twinx()
    ax2.set_ylabel("ASR")
    line2, = ax2.plot(np.arange(1, int(len(asrs)+1)), asrs, label="ASR",color="red")
    ax2.tick_params(axis="y")

    plt.title("Global Accuracy and ASR Over Training Rounds")
    fig.tight_layout()
    plt.grid(True)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    plt.grid(True, axis='both')
    plt.savefig(file+'GlobalAccAsr.pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    #Loss
    plt.plot(losses, label="Loss")
    plt.xlabel("Training Rounds")
    plt.ylabel("Loss")
    plt.title("Global Loss Over Training Rounds")
    plt.grid(True, axis='both')

    plt.savefig(file + 'GlobalLoss.pdf')
    if verbose:
        plt.show()
    plt.close()
    
def perTrainingRoundPlots(array,epochs, Metric,file="",verbose=True):
    """
    Plot figures for per training round data
    :param array: Values to plot (Accs/losses etc)
    :param epochs: Number of local training epochs for presentation in plot
    :param Metric: Name of metric plotting (acc/asr etc)
    :param file: Output file
    :param verbose: Verbose toggle
    :return: Plots for a paramter at every training round
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    # Plot the accuracy over epochs
    for i, modelVals in enumerate(array):
        modelVals = np.array(modelVals).flatten()
        N = len(modelVals)
        plt.plot(np.arange(1, len(modelVals) + 1), modelVals, label=f'Client {i + 1}')
    # Highlight communication rounds
    for i in range(epochs, N + 1, epochs):
        plt.axvline(x=i, color='red', linestyle='--')
    # Labels and title
    plt.xlabel("Epochs")
    plt.ylabel(Metric)
    plt.title("Client " + Metric + " Over Epochs")
    plt.legend()
    plt.grid(True, axis='both')
    # Show plot
    plt.savefig(file + "Client" + Metric + ".pdf")
    if verbose:
        plt.show()
    plt.close()
    plt.figure(figsize=(10, 6))
    # Plot the accuracy over epochs
    for i, modelVals in enumerate(array):
        modelVals = np.array(modelVals).flatten()
        plt.plot(np.arange(1, len(modelVals[::epochs]) + 1), modelVals[::epochs], label=f'Client {i + 1}')
    # Labels and title
    plt.xlabel("Training Rounds")
    plt.ylabel(Metric)
    plt.title("Client " + Metric + " Over Training Rounds")
    plt.legend()
    plt.grid(True, axis='both')
    # Show plot
    plt.savefig(file + "Client" + Metric + "pRound.pdf")
    if verbose:
        plt.show()
    plt.close()

def plotMI(file,numClients):
    """
    Plots BSCI value at every global training round
    :param file: Output file
    :param numClients: Number of clients in system
    :return: Inference plots
    """
    data = []
    fileData = file + "preds.csv"
    with open(fileData, 'r', newline='') as fileData:
        reader = csv.reader(fileData)
        for row in reader:
            data.append(row)
    data = data[1:]
    data = [[float(x) for x in row] for row in data]
    data_by_model = list(zip(*data))

    epochs = list(range(1, len(data) + 1))
    plt.figure(figsize=(10, 6))

    for i, miSuccess in enumerate(data_by_model):
        if i == numClients:
            plt.plot(epochs, miSuccess, label=f'Federated Model', linestyle='--')
        else:
            plt.plot(epochs, miSuccess, label=f'Model {i + 1}')
    plt.xlabel('Training Rounds')
    plt.ylabel('Backdoor Inference Success')
    plt.title('Backdoor Inference Success Over Training Rounds')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.savefig(file + "plots/InferenceSuccess.pdf")
    plt.show()
    plt.close()
    plt.plot(epochs, data_by_model[numClients])
    plt.xlabel('Training Rounds')
    plt.ylabel('Inference Success')
    plt.title('Inference Success Over Training Rounds')
    plt.grid(True,axis='both')
    plt.savefig(file + "plots/InferenceSuccessGlbal.pdf")
    plt.show()
    plt.close()

def plotSelected(file,selected,asr,mi,window = 0):
    """
    For RA schemes that select a subset of clients, plot whether the client was selected or not
    :param file: Output file
    :param selected: Array displaying whether a malicious client was selected per round
    :param asr: ASR at every round
    :param mi: MI success at every round
    :param window: Window to smooth plot if required
    :return: Plots for output file
    """
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy and ASR on the primary y-axis
    line1, = ax1.plot(np.arange(1,len(asr)+1), asr, label='ASR')
    if window > 0:
        mi = np.convolve(mi, np.ones(window)/window, mode='same')
        line2, = ax1.plot(np.arange(1,len(mi)+1), mi, label='Inference Success')
    else:
        line2, = ax1.plot(np.arange(1, len(mi) + 1), mi, label='Inference Success')
    ax1.set_xlabel('Training Rounds')
    ax1.set_ylabel('ASR / Inference Success')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    line3, = ax2.plot(np.arange(1,len(selected)+1), selected, 'r', label='Model Selected')
    ax2.set_ylabel('Malicious client selected')
    ax2.tick_params(axis='y')

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.tight_layout()
    ax1.grid(True, axis='both')
    try:
        plt.savefig(file + "plots/ASRMI" + file.split("/")[0] + file.split("/")[1] + file.split("/")[2] + ".pdf",
                    dpi=500)
    except:
        try:
            plt.savefig(file + "plots/ASRMI" + file.split("/")[0] + file.split("/")[1] + ".pdf",dpi=500)
        except:
            plt.savefig(file + "plots/ASRMI" + file.split("/")[0] + ".pdf", dpi=500)
    plt.show()
    plt.close()
def ClientPlots(accs,losses,epochs,file="",verbose=True):
    """
    Plot all figures for a client
    :param accs: Array of accuracies
    :param losses: Array of losses
    :param epochs: Number of local epochs
    :param file: Output file
    :param verbose: Verbose Toggle
    :return: All client plots saved to output file
    """
    perTrainingRoundPlots(accs,epochs, "Accuracy",file,verbose=verbose)
    perTrainingRoundPlots(losses,epochs, "Loss",file,verbose=verbose)

def AIPlots(gAccs,gASRs,gLosses,accs,losses,epochs,file="",verbose=True):
    """
    Plot all figures for the global model and client models
    :param gaccs: Array of accuracies
    :param gASRs: Array of ASR's
    :param glosses: Array of losses
    :param accs: Array of accuracies for clients
    :param losses: Array of losses for clients
    :param epochs: Number of local epochs
    :param file: Output file
    :param verbose: Verbose Toggle
    :return: All client plots saved to output file
    """
    GlobalPlots(gAccs,gASRs,gLosses,file,verbose=verbose)
    ClientPlots(accs,losses,epochs,file,verbose=verbose)
