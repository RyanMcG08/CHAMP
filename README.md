# CHAMP ARTIFACT
Federated Learning framework allowing the user to implement the champ algorithm

## Overview
This tool allows you to simulate:
- Federated learning across multiple clients
- Backdoor attacks (with different trigger types)
- Defense mechanisms like median and trimmed mean and more
- FL systems under attack by backdoor poisoning, with and without the adaptive attack champ

## Supplementary Demonstrations attached
In this artifact we have attached a visualisation of our attack and for comparison a visualisation of a baseline backdoor poisoning attack, as two mp4 videos. Here, the malicious update is highlighted as a star and clients that have been included in the multi-krum robust 
aggregation scheme are highlighted in green. We can see our attack moving into the distribution of accepted updates by 
the RA scheme, whereas a naive attacker cannot enter the set of accepted updates to the global model and therefore implements no poisoning.

These experiments were completed on the Cifar-10 dataset using Multi-Krum as the RA scheme. Using the same setup as highlighted in the main text of the paper. The comparison is shown in the attached file "ComparisonOfAttacks.mp4".

## How to Run

Run the main script with any combination of arguments:

python RunAttack.py [arguments]

Baseline Example:

python RunAttack.py --trainingRounds 50 --numClients 10 --numMal 1 --scheme 1 --adaptive 1 --attack_type 0

## Overview of arguements

Arguments:

- trainingRounds (int, default: 50): Number of global training rounds  
- numClients (int, default: 10): Total number of clients  
- numMal (int, default: 1): Number of malicious clients  
- epochs (int, default: 5): Local training epochs per client  
- headerFile (str, default: "test"): Path for logging output files  
- verbose (int, 0 or 1, default: 0): Verbose mode toggle  
- scheme (int, default: 0): Aggregation/defense scheme ID  
- param (float, default: 0.0): Parameter for selected scheme if possible  
- adaptive (int, 0 or 1, default: 0): Toggle for adaptive loss function  
- r (int, default: 5): number of training rounds to consider in $\alpha$ calculations  
- ai (int, default: 1): Number of intervals we recalculate $\alpha$
- attack_type (int, default: 0): Attack type
- asr (int, 0 or 1, default: 0): Toggle for using ASR as loss function scalar instead of Membership Inference  
- percentages (list of floats, default: [0.3, 0.2, 0.1, 0.0, 0.0, 0.0]): Percentage of poisoned samples and number of reference models 
- cleanTog (int, 0 or 1, default: 1): Toggle cleaning all models used in simulation of FL system
- net (str, choices: "alexnet", "fashionMNISTCNN", default: "fashionMNISTCNN"): Model architecture  
- dataset (str, choices: "MNIST", "cifar10", "fashionMNIST", default: "fashionMNIST"): Dataset name  
- backdoor (str, choices: "one", "three", "five", default: "letterR"): Backdoor trigger type  
- alpha (int, default: 0): Parameter for Dirichlet distribution in non-IID data)
- lossFunc (int, default: 0): Loss function metric 
- lr (float, default: 0.1): Local client learning rate

## Further Details on Arguments:
#### Param
0. FedAvg 
1. Median 
2. Trimmed-Mean
3. Krum/Mkrum
4. Bulyan
5. RFA
6. Direction Alignment Inspection
7. RLR
8. FoolsGold

#### Attack_type
0. Targeted Backdoor attack
1. Untargeted Backdoor attack

#### Loss
1. Euclidean Distance
2. Huber Loss
3. Cosine Similarity

## Supported Models & Datasets

Models:
- alexnet
- fashionMNISTCNN

Datasets:
- cifar10
- fashion-mnist

## Backdoor Trigger Types

one: Injects a 1x1 trigger  
three: Injects a 3x3 trigger  
five: Injects a 5x5 trigger  
letterR: Injects a Letter R trigger

## Output

- Training logs, plots and metrics saved to the file --headerFile
- Use --verbose 1 for detailed output during training
- use --cleanTog 0 to keep all interim files (local models and reference models)
- Training logs and metrics saved to the file --headerFile
- Use --verbose 1 for detailed output during training
- use --cleanTog 0 to save all interim files (local models and reference models)