import torch
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os


import random

# It is important to know the superclasses, we are going to do it only with the CIFAR100 classes
# List of superclasses
superclass_labels = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2"
]

# Map class index to its superclass
class_to_superclass = {
    0: "aquatic mammals", 1: "aquatic mammals", 2: "aquatic mammals", 3: "aquatic mammals", 4: "aquatic mammals", 5: "fish", 6: "fish", 7: "fish", 8: "fish", 9: "fish", 10: "flowers", 11: "flowers", 12: "flowers", 13: "flowers", 14: "flowers",
    15: "food containers", 16: "food containers", 17: "food containers", 18: "food containers", 19: "food containers", 20: "fruit and vegetables", 21: "fruit and vegetables", 22: "fruit and vegetables", 23: "fruit and vegetables", 24: "fruit and vegetables",
    25: "household electrical devices", 26: "household electrical devices", 27: "household electrical devices", 28: "household electrical devices", 29: "household electrical devices", 30: "household furniture", 31: "household furniture", 32: "household furniture", 33: "household furniture", 34: "household furniture",
    35: "insects", 36: "insects", 37: "insects", 38: "insects", 39: "insects", 40: "large carnivores", 41: "large carnivores", 42: "large carnivores", 43: "large carnivores", 44: "large carnivores", 45: "large man-made outdoor things", 46: "large man-made outdoor things", 47: "large man-made outdoor things", 48: "large man-made outdoor things", 49: "large man-made outdoor things",
    50: "large natural outdoor scenes", 51: "large natural outdoor scenes", 52: "large natural outdoor scenes", 53: "large natural outdoor scenes", 54: "large natural outdoor scenes", 55: "large omnivores and herbivores", 56: "large omnivores and herbivores", 57: "large omnivores and herbivores", 58: "large omnivores and herbivores", 59: "large omnivores and herbivores",
    60: "medium-sized mammals", 61: "medium-sized mammals", 62: "medium-sized mammals", 63: "medium-sized mammals", 64: "medium-sized mammals", 65: "non-insect invertebrates", 66: "non-insect invertebrates", 67: "non-insect invertebrates", 68: "non-insect invertebrates", 69: "non-insect invertebrates", 70: "people", 71: "people", 72: "people", 73: "people", 74: "people", 75: "reptiles", 76: "reptiles", 77: "reptiles", 78: "reptiles", 79: "reptiles",
    80: "small mammals", 81: "small mammals", 82: "small mammals", 83: "small mammals", 84: "small mammals", 85: "trees", 86: "trees", 87: "trees", 88: "trees", 89: "trees", 90: "vehicles 1", 91: "vehicles 1", 92: "vehicles 1", 93: "vehicles 1", 94: "vehicles 1", 95: "vehicles 2", 96: "vehicles 2", 97: "vehicles 2", 98: "vehicles 2", 99: "vehicles 2"
}


def obtain_superclasses(labels):

    labels_copied = np.array(labels)

    super_class_dictionary = dict() # Each entry contains a list of classes whose index its theirs superclass
    super_class_names_list = [] # Each entry contains a pair superclass_name, [int_1,int_2], where int_2 is the number of classes whose
                                #superclass is superclass_name and int_1 will be used later as an iterator for the list on the dicticionary
    
    while(len(labels_copied)>1):
        class_id = labels_copied[0]
        super_class_name = class_to_superclass[class_id]
        
        
        grouped_superclass = labels_copied[[class_to_superclass[label] == super_class_name for label in labels_copied]]

        class_names = np.unique(grouped_superclass)

        random.shuffle(class_names)
        super_class_names_list.append([super_class_name,[0,len(class_names)]])
        super_class_dictionary[super_class_name] = class_names

        labels_copied = labels_copied[[class_to_superclass[label] != super_class_name for label in labels_copied]]

    random.shuffle(super_class_names_list)

    return super_class_dictionary,super_class_names_list

def do_ood_folds(dataset_name, X,y, K, data_path, seed=42):

    print("Obtaining superclasses...",flush=True)
    super_class_dictionary, super_class_names_list = obtain_superclasses(y)

    print("Selecting OOD classes...",flush=True)
    n_class_ODD = int(0.2*len(np.unique(y)))
    OOD_classes = []
    iterator = 0
    while(len(OOD_classes) < n_class_ODD):
        if (iterator == len(super_class_names_list)):
            iterator=0
        
        while(super_class_names_list[iterator][1][0] == super_class_names_list[iterator][1][1]):
            iterator = iterator + 1
        
        OOD_classes.append(super_class_dictionary[super_class_names_list[iterator][0]][super_class_names_list[iterator][1][0]])
        super_class_names_list[iterator][1][0] = super_class_names_list[iterator][1][0] + 1
        iterator = iterator + 1
   
    print(OOD_classes, flush=True)
    OOD_image =[img for img, label in zip(X, y) if label in OOD_classes]
    OOD_label =[img for img, label in zip(y, y) if label in OOD_classes]

    ID_image =[img for img, label in zip(X, y) if label not in OOD_classes]
    ID_label =[img for img, label in zip(y, y) if label not in OOD_classes]
    

    print("Creating ID folds", flush=True)
    sfolds = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sfolds.split(ID_image, ID_label)):
        # Create sets of train and test
        train_data = [(ID_image[i], ID_label[i]) for i in train_idx]
        test_data = [(ID_image[i], ID_label[i]) for i in val_idx]

        try:
            os.makedirs(f"{data_path}/{dataset_name}_{seed}", exist_ok=False)
        except:
            "Be carefull directory already created"
        # Save each fold
        torch.save(train_data, f"{data_path}/{dataset_name}_{seed}/train_ID_fold_{fold}.pt")
        torch.save(test_data, f"{data_path}/{dataset_name}_{seed}/test_ID_fold_{fold}.pt")

        print(f"Fold {fold} saved: {len(train_data)} train, {len(test_data)} test")

    print("ID Folds saved", flush=True)

    print("Creating OOD folds", flush=True)
    super_class_dictionary_OOD, super_class_names_OOD = obtain_superclasses(OOD_label)

    n_class_ODD = int(len(super_class_names_OOD)/K)
    folds_classes = [ [] for _ in range(K)]
    print(n_class_ODD, flush=True)
    print(len(super_class_names_OOD), flush=True)
    fold_iterator = 0
    iterator = 0

    while(len(folds_classes[0]) < n_class_ODD):
        if (iterator == len(super_class_names_OOD)):
            iterator=0
        fold_iterator = 0
        while(fold_iterator!=K):
            while(super_class_names_OOD[iterator][1][0] == super_class_names_OOD[iterator][1][1]):
                iterator = iterator + 1
            
            while( (super_class_names_OOD[iterator][1][0] < super_class_names_OOD[iterator][1][1]) and fold_iterator!=K):
                folds_classes[fold_iterator].append(super_class_dictionary_OOD[super_class_names_OOD[iterator][0]][super_class_names_OOD[iterator][1][0]])
                super_class_names_OOD[iterator][1][0] = super_class_names_OOD[iterator][1][0] + 1
                fold_iterator = fold_iterator + 1
                
            iterator = iterator + 1
    
    for fold, fold_classes in enumerate(folds_classes):
        if len(fold_classes) == 0:
            raise IOError('test index vacio')
        
        train_image =[img for img, label in zip(OOD_image, OOD_label) if label not in fold_classes]
        train_label =[img for img, label in zip(OOD_label, OOD_label) if label not in fold_classes]

        test_image =[img for img, label in zip(OOD_image, OOD_label) if label in fold_classes]
        test_label =[img for img, label in zip(OOD_label, OOD_label) if label in fold_classes]

        train_data = [(train_image[i], train_label[i]) for i in range(len(train_image))]
        test_data = [(test_image[i], test_label[i]) for i in range(len(test_image))]

        # Save each fold
        torch.save(train_data, f"{data_path}/{dataset_name}_{seed}/train_OOD_fold_{fold}.pt")
        torch.save(test_data, f"{data_path}/{dataset_name}_{seed}/test_OOD_fold_{fold}.pt")

        print(f"Fold {fold} saved: {len(train_data)} train, {len(test_data)} test")

    print("OOD Folds saved", flush=True)


def main():
    # Configuración
    K = 5  # Número de folds
    data_path = "./data"  # Ruta para guardar los folds
    seeds = [1846782, 620738, 326040, 1466620, 981127, 1291430, 540103, 723262, 1531415]
    random.seed(42)

    cifar_100_train = CIFAR100(root=data_path, train=True, download=False)
    cifar_100_test = CIFAR100(root=data_path, train=False, download=False)

    X_Cifar100_train = [img for img, _ in cifar_100_train]
    y_Cifar100_train = [label for _, label in cifar_100_train]

    X_Cifar100_test = [img for img, _ in cifar_100_test]
    y_Cifar100_test = [label for _, label in cifar_100_test]

    X_Cifar100 = X_Cifar100_train + X_Cifar100_test
    y_Cifar100 = y_Cifar100_train + y_Cifar100_test

    list_X = [X_Cifar100]
    list_y = [y_Cifar100]
    list_names = ["super_cifar100"]

    for seed in seeds:
        random.seed(seed)
        for i in range(len(list_X)):
            print(f"Creating ood folds to {list_names[i]}")
            do_ood_folds(list_names[i],list_X[i],list_y[i],5,"./folds", seed)

if __name__ == '__main__':
    
    main()