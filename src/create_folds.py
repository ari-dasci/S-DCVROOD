import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, DTD
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import os
from torchvision.datasets import ImageFolder

def do_folds_group(dataset_name, X,y, K, data_path, seed=42):

    # Aply Stratified K-Fold
    sfolds = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sfolds.split(X, y, y)):
        # Create sets of train and test
        train_data = [(X[i], y[i]) for i in train_idx]
        test_data = [(X[i], y[i]) for i in val_idx]

        try:
            os.makedirs(f"{data_path}/{dataset_name}_{seed}", exist_ok=False)
        except:
            "Be carefull directory already created"
        # Save each fold
        torch.save(train_data, f"{data_path}/{dataset_name}_{seed}/train_group_fold_{fold}.pt")
        torch.save(test_data, f"{data_path}/{dataset_name}_{seed}/test_group_fold_{fold}.pt")

        print(f"Fold {fold} saved: {len(train_data)} train, {len(test_data)} test", flush=True)

    print("Folds saved", flush=True)

def do_folds_strated(dataset_name, X,y, K, data_path, seed=42):

    # Aply Stratified K-Fold
    sfolds = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sfolds.split(X, y)):
        # Create sets of train and test
        train_data = [(X[i], y[i]) for i in train_idx]
        test_data = [(X[i], y[i]) for i in val_idx]

        try:
            os.makedirs(f"{data_path}/{dataset_name}_{seed}", exist_ok=False)
        except:
            "Be carefull directory already created"

        # Save each fold
        torch.save(train_data, f"{data_path}/{dataset_name}_{seed}/train_strat_fold_{fold}.pt")
        torch.save(test_data, f"{data_path}/{dataset_name}_{seed}/test_strat_fold_{fold}.pt")

        print(f"Fold {fold} saved: {len(train_data)} train, {len(test_data)} test", flush=True)

    print("Folds saved", flush=True)

def main():

    # Configuración
    K = 5  # Number of folds
    data_path = "./data"  # folder to save the data
    seeds = [42, 1846782, 620738, 326040, 1466620, 981127, 1291430, 540103, 723262, 1531415] # Seeds for randomness and reproducibility
    cifar_10_train = CIFAR10(root=data_path, train=True, download=False)
    cifar_10_test = CIFAR10(root=data_path, train=False, download=False)

    
    X_Cifar10_train = [img for img, _ in cifar_10_train]
    y_Cifar10_train = [label for _, label in cifar_10_train]

    X_Cifar10_test = [img for img, _ in cifar_10_test]
    y_Cifar10_test = [label for _, label in cifar_10_test]

    X_Cifar10 = X_Cifar10_train + X_Cifar10_test
    y_Cifar10 = y_Cifar10_train + y_Cifar10_test
    print(f"Cifars loaded", flush=True)

    cifar_100_train = CIFAR100(root=data_path, train=True, download=False)
    cifar_100_test = CIFAR100(root=data_path, train=False, download=False)

    X_Cifar100_train = [img for img, _ in cifar_100_train]
    y_Cifar100_train = [label for _, label in cifar_100_train]

    X_Cifar100_test = [img for img, _ in cifar_100_test]
    y_Cifar100_test = [label for _, label in cifar_100_test]

    X_Cifar100 = X_Cifar100_train + X_Cifar100_test
    y_Cifar100 = y_Cifar100_train + y_Cifar100_test

    
    mnist_train = EMNIST(root='./data', split='mnist', train=True, download=False)
    mnist_test = EMNIST(root='./data', split='mnist', train=False, download=False)

    X_Mnist_train = [img for img, _ in mnist_train]
    y_Mnist_train = [label for _, label in mnist_train]

    X_Mnist_test = [img for img, _ in mnist_test]
    y_Mnist_test = [label for _, label in mnist_test]

    X_mnist = X_Mnist_train + X_Mnist_test
    y_mnist = y_Mnist_train + y_Mnist_test
    print(f"Mnist loaded", flush=True)

    
    emnist_train = EMNIST(root='./data', split='letters', train=True, download=False)
    emnist_test = EMNIST(root='./data', split='letters', train=False, download=False)

    X_eMnist_train = [img for img, _ in emnist_train]
    y_eMnist_train = [label for _, label in emnist_train]

    X_eMnist_test = [img for img, _ in emnist_test]
    y_eMnist_test = [label for _, label in emnist_test]

    X_emnist = X_eMnist_train + X_eMnist_test
    y_emnist = y_eMnist_train + y_eMnist_test
    print(f"Letters loaded", flush=True)

    
    dtd_train = DTD(root='./data', split='train', download=False)
    dtd_test = DTD(root='./data', split='test', download=False)

    X_DTD_train = [img for img, _ in dtd_train]
    y_DTD_train = [label for _, label in dtd_train]

    X_DTD_test = [img for img, _ in dtd_test]
    y_DTD_test = [label for _, label in dtd_test]

    X_dtd = X_DTD_train + X_DTD_test
    y_dtd = y_DTD_train + y_DTD_test
    print(f"Textures loaded", flush=True)
    
    tiny_imagenet_train = ImageFolder(root=f'./data/tiny-imagenet-200/train')
    tiny_imagenet_test = ImageFolder(root=f'./data/tiny-imagenet-200/test')

    X_tiny_train = [img for img, _ in tiny_imagenet_train]
    y_tiny_train = [label for _, label in tiny_imagenet_train]

    X_tiny_test = [img for img, _ in tiny_imagenet_test]
    y_tiny_test = [label for _, label in tiny_imagenet_test]

    X_tiny = X_tiny_train + X_tiny_test
    y_tiny = y_tiny_train + y_tiny_test
    print(f"Imagenet_loaded", flush=True)

    list_X = [X_Cifar10, X_Cifar100, X_mnist, X_emnist, X_dtd, X_tiny]
    list_y = [y_Cifar10, y_Cifar100, y_mnist, y_emnist, y_dtd, y_tiny]
    list_names = ["cifar10", "cifar100", "mnist", "letters", "dtd", "tiny_imagenet_200"]
    print(f"Datasets combined", flush=True)
    for seed in seeds:
        for i in range(len(list_X)):
            print(f"Creating strat folds to {list_names[i]}", flush=True)
            do_folds_strated(list_names[i],list_X[i],list_y[i],5,"./folds", seed)
            print(f"Creating group folds to {list_names[i]}", flush=True)
            do_folds_group(list_names[i],list_X[i],list_y[i],5,"./folds", seed)

if __name__ == '__main__':
    
    main()