import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, DTD, ImageFolder
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
import os
import argparse



import random




def do_ood_folds(dataset_name, X,y, K, data_path, seed=42, number=0):

    print("Selecting OOD classes...",flush=True)
    n_class_ODD = int(0.2*len(np.unique(y)))
    OOD_classes = random.sample(range(0, 100), n_class_ODD)
    OOD_image =[img for img, label in zip(X, y) if label in OOD_classes]
    OOD_label =[img for img, label in zip(y, y) if label in OOD_classes]

    ID_image =[img for img, label in zip(X, y) if label not in OOD_classes]
    ID_label =[img for img, label in zip(y, y) if label not in OOD_classes]

    print("Creating ID fold", flush=True)
    sfolds = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sfolds.split(ID_image, ID_label))

    train_images = [ID_image[i] for i in train_idx]
    train_labels = [ID_label[i] for i in train_idx]
    test_data = [(ID_image[i], ID_label[i]) for i in test_idx]

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    train_data = [(X_train[i], y_train[i]) for i in range(0,len(X_train))]
    val_data = [(X_val[i], y_val[i]) for i in range(0,len(X_val))]

    try:
        os.makedirs(f"{data_path}/{dataset_name}", exist_ok=False)
    except:
        "Be carefull directory already created"

    torch.save(train_data, f"{data_path}/{dataset_name}/train_ID_fold_{number}.pt")
    torch.save(val_data, f"{data_path}/{dataset_name}/val_ID_fold_{number}.pt")
    torch.save(test_data, f"{data_path}/{dataset_name}/test_ID_fold_{number}.pt")

    print(f"Fold saved: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    print("ID Fold saved", flush=True)

    print("Creating OOD fold", flush=True)
    gfolds = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
    train_idx, test_idx = next(gfolds.split(OOD_image, OOD_label, OOD_label))

    train_images = [OOD_image[i] for i in train_idx]
    train_labels = [OOD_label[i] for i in train_idx]
    test_data = [(OOD_image[i], OOD_label[i]) for i in test_idx]

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    train_data = [(X_train[i], y_train[i]) for i in range(0,len(X_train))]
    val_data = [(X_val[i], y_val[i]) for i in range(0,len(X_val))]

    torch.save(train_data, f"{data_path}/{dataset_name}/train_OOD_fold_{number}.pt")
    torch.save(val_data, f"{data_path}/{dataset_name}/val_OOD_fold_{number}.pt")
    torch.save(test_data, f"{data_path}/{dataset_name}/test_OOD_fold_{number}.pt")

    print(f"Fold saved: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

def do_folds_group(dataset_name, X,y, K, data_path, seed=42, number=0):

    print("Creating group fold", flush=True)
    # Aply Stratified K-Fold
    gfolds = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
    train_idx, test_idx = next(gfolds.split(X, y, y))

    try:
        os.makedirs(f"{data_path}/{dataset_name}", exist_ok=False)
    except:
        "Be carefull directory already created"

    train_images = [X[i] for i in train_idx]
    train_labels = [y[i] for i in train_idx]
    test_data = [(X[i], y[i]) for i in test_idx]

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    train_data = [(X_train[i], y_train[i]) for i in range(0,len(X_train))]
    val_data = [(X_val[i], y_val[i]) for i in range(0,len(X_val))]

    torch.save(train_data, f"{data_path}/{dataset_name}/train_group_fold_{number}.pt")
    torch.save(val_data, f"{data_path}/{dataset_name}/val_group_fold_{number}.pt")
    torch.save(test_data, f"{data_path}/{dataset_name}/test_group_fold_{number}.pt")

    print(f"Fold saved: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

def do_folds_strated(dataset_name, X,y, K, data_path, seed=42, number=0):

    # Aply Stratified K-Fold
    sfolds = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sfolds.split(X, y))

    try:
        os.makedirs(f"{data_path}/{dataset_name}", exist_ok=False)
    except:
        "Be carefull directory already created"

    train_images = [X[i] for i in train_idx]
    train_labels = [y[i] for i in train_idx]
    test_data = [(X[i], y[i]) for i in test_idx]

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    train_data = [(X_train[i], y_train[i]) for i in range(0,len(X_train))]
    val_data = [(X_val[i], y_val[i]) for i in range(0,len(X_val))]

    torch.save(train_data, f"{data_path}/{dataset_name}/train_strat_fold_{number}.pt")
    torch.save(val_data, f"{data_path}/{dataset_name}/val_strat_fold_{number}.pt")
    torch.save(test_data, f"{data_path}/{dataset_name}/test_strat_fold_{number}.pt")

    print(f"Fold saved: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")


def main(args):
    seed = args.seed
    number = args.number
    # Configuración
    K = 5  # Número de folds
    data_path = "./data"  # Ruta para cargar los datasets
    random.seed(seed) 

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

    list_X = [X_dtd, X_tiny, X_emnist, X_mnist, X_Cifar100, X_Cifar10]
    list_y = [y_dtd, y_tiny, y_emnist, y_mnist, y_Cifar100, y_Cifar10]
    list_names = ["dtd", "tiny_imagenet_200", "letters", "mnist", "cifar100", "cifar10"]


    for i in range(len(list_X)):
        print(f"Creating strat fold to {list_names[i]}")
        do_folds_strated(list_names[i],list_X[i],list_y[i],K,"./gt_folds",seed, number)
        print(f"Creating group fold to {list_names[i]}")
        do_folds_group(list_names[i],list_X[i],list_y[i],K,"./gt_folds",seed, number)

    do_ood_folds("super_cifar100", X_Cifar100, y_Cifar10, K, "./gt_folds", seed, number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create folds')
    parser.add_argument('--seed', required=True, type=int, help='Seed')
    parser.add_argument('--number', required=True, type=int, help='Numbered_fold')
    args = parser.parse_args()

    main(args)