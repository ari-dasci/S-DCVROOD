from types import SimpleNamespace

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
from tqdm.auto import tqdm
import argparse
import sys
import os
import numpy as np
import json

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from trainer import ClassificationModel
from src.dataset_loader import ClassificationDataset

def process_and_save(model, dataloader, save_path, mode):
    model.eval()
    logits_list = []
    embs_list = []
    label_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Extracting : {mode}', position=0, leave=True):
            images, labels = batch
            data = images.cuda()
            data = data.float()

            logits, embs = model(data, return_feature=True)

            

            logits_list = logits_list + logits.cpu().numpy().tolist()
            embs_list = embs_list + embs.cpu().numpy().tolist()
            label_list =  label_list + labels.cpu().numpy().tolist()

    df_logits = pd.DataFrame(logits_list)
    df_logits.columns = [f'logits{i}' for i in range(df_logits.shape[1])]

    df_embs = pd.DataFrame(embs_list)
    df_embs.columns = [f'embeddings{i}' for i in range(df_embs.shape[1])]

    df_labels = pd.DataFrame(label_list, columns=['labels'])

    # Concatenar todo en un solo DataFrame
    df_total = pd.concat([df_logits, df_embs, df_labels], axis=1)

    df_total.to_csv(f"{save_path}/{mode}.csv", index=False)
    
def obtain_data(X,y, hyperparameters, transform, X_outliner=None, y_outliner=None, transform_2 = None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if X_outliner is not None:
        X_train_out, X_val_out, y_train_out, y_val_out = train_test_split(X_outliner, y_outliner, test_size=0.2, stratify=y_outliner, random_state=42)
    else:
        X_train_out = None
        y_train_out = None
        X_val_out = None
        y_val_out = None
        
    # Create DataLoaders
    train_dataset = OutlinerDataset(X_train,y_train, X_train_out, y_train_out, transforms=transform, transforms_2=transform_2)
    val_dataset = OutlinerDataset(X_val,y_val, X_val_out, y_val_out, transforms=transform, transforms_2=transform_2)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, num_workers=16, pin_memory=False, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["val_batch_size"], shuffle=False, num_workers=16, pin_memory=False, persistent_workers=True)

    return train_loader, val_loader, y_train

def main(args):

    torch.manual_seed(2809)

    data_trainded = args.id_database
    data_path = args.database
    experiment_name = args.experiment_name
    output_dir = args.output_dir
    fold = args.fold
    epoch = args.epoch
    version = args.version
    seed = args.seed
    calculate_val = args.val
    

    batch_size = 64
    torch.set_float32_matmul_precision('medium')
    device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(6)

    results_dir = os.path.join(output_dir, f"{experiment_name}_{data_trainded}_version_{version}_fold_{fold}_{seed}")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 64

    if "mnist" in data_path or "letters" in data_path:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transform])
    else:
        transform = transform

    if experiment_name == "resnet18":
        transform = transforms.Compose([transform, transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])]) 
    elif experiment_name == "efficientnet_l":
        batch_size = 32
        transform = transforms.Compose([transform, transforms.Resize((480, 480)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    elif experiment_name == "vit_b_16" or experiment_name == "gt_vit_b_16":
        batch_size = 64
        transform = transforms.Compose([transform, transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        print("ERROR UNRECOGNISED MODEL")
        sys.exit(-1)

    # Load model
    if epoch!=None:
        model_path = os.path.join("models", f"{experiment_name}_{data_trainded}_fold_{fold}_{seed}", f"version_{version}", "checkpoints", f"{experiment_name}_{data_trainded}_fold_{fold}_{seed}_epoch={epoch}-val_loss=0.00.ckpt")
    else:
        checkpoints_dir = os.path.join("models", f"{experiment_name}_{data_trainded}_fold_{fold}_{seed}", f"version_{version}", "checkpoints")
        checkpoints_names = list(os.listdir(checkpoints_dir))
        model_path = os.path.join(checkpoints_dir, checkpoints_names[-1])

    model = ClassificationModel.load_from_checkpoint(model_path)

    print("Model loaded",flush=True)
    model.to(device)
    model.eval()

    database_path = os.path.join("folds", data_path)

    if not calculate_val:
        if data_path == "super_cifar100":
            id_data = torch.load(f"{database_path}_{seed}/test_ID_fold_{fold}.pt")
            train_data = torch.load(f"{database_path}_{seed}/train_ID_fold_{fold}.pt")
            val_data = torch.load(f"{database_path}_{seed}/val_ID_fold_{fold}.pt")
            ood_data = torch.load(f"{database_path}_{seed}/test_OOD_fold_{fold}.pt")
            ood_train_data = torch.load(f"{database_path}_{seed}/train_OOD_fold_{fold}.pt")
            ood_val_data = torch.load(f"{database_path}_{seed}/val_OOD_fold_{fold}.pt")
        else:
            if data_trainded == data_path:
                id_data = torch.load(f"{database_path}_{seed}/test_strat_fold_{fold}.pt")
                train_data = torch.load(f"{database_path}_{seed}/train_strat_fold_{fold}.pt")
                val_data = torch.load(f"{database_path}_{seed}/val_strat_fold_{fold}.pt")
            else:
                ood_data = torch.load(f"{database_path}_{seed}/test_group_fold_{fold}.pt")
                ood_train_data = torch.load(f"{database_path}_{seed}/train_group_fold_{fold}.pt")
                ood_val_data = torch.load(f"{database_path}_{seed}/val_group_fold_{fold}.pt")
    
    else:
        if data_path == "super_cifar100":
            id_data = torch.load(f"{database_path}_{seed}/test_ID_fold_{fold}.pt")
            ood_data = torch.load(f"{database_path}_{seed}/test_OOD_fold_{fold}.pt")
            train_data = torch.load(f"{database_path}_{seed}/train_ID_fold_{fold}.pt")
            ood_train_data = torch.load(f"{database_path}_{seed}/train_OOD_fold_{fold}.pt")
        else:
            if data_trainded == data_path:
                id_data = torch.load(f"{database_path}_{seed}/test_strat_fold_{fold}.pt")
                train_data = torch.load(f"{database_path}_{seed}/train_strat_fold_{fold}.pt")
            else:
                ood_train_data = torch.load(f"{database_path}_{seed}/train_group_fold_{fold}.pt")
                ood_data = torch.load(f"{database_path}_{seed}/test_group_fold_{fold}.pt")

    
    print("Data Loaded",flush=True)
    # Convert to tensors
    if data_trainded == data_path:
        if not calculate_val:
            X_train = [img for img, _ in train_data]
            X_val = [img for img, _ in val_data]
            y_train = [label for _, label in train_data]
            y_val = [label for _, label in val_data]

            if data_path == "super_cifar100":
                auxliary = np.arange(0, len(np.unique(y_train)))
                maping = {key: swap for key,swap in zip(np.unique(y_train),auxliary)}
                y_train = [maping[i] for i in y_train]
                y_val = [maping[i] for i in y_val]

        else:
            X = [img for img, _ in train_data]
            y = [label for _, label in train_data]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            if data_path == "super_cifar100":
                auxliary = np.arange(0, len(np.unique(y_train)))
                maping = {key: swap for key,swap in zip(np.unique(y_train),auxliary)}
                y_train = [maping[i] for i in y_train]
                y_val = [maping[i] for i in y_val]

        train_dataset = ClassificationDataset(X_train,y_train,transforms=transform)
        val_dataset = ClassificationDataset(X_val,y_val,transforms=transform)


    # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

        X_id = [img for img, _ in id_data]
        y_id = [label for _, label in id_data]
        if data_path == "super_cifar100":
            y_id = [maping[i] for i in y_id]
        ID_dataset = ClassificationDataset(X_id,y_id,transforms=transform)
        id_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        
        if data_path == "super_cifar100":
            X_ood = [img for img, _ in ood_data]
            y_ood = [label for _, label in ood_data]
            OOD_dataset = ClassificationDataset(X_ood,y_ood,transforms=transform)

            if not calculate_val:
                X_ood_train = [img for img, _ in ood_train_data]
                y_ood_train = [label for _, label in ood_train_data]

                X_ood_val = [img for img, _ in ood_val_data]
                y_ood_val = [label for _, label in ood_val_data]

                OOD_dataset_train = ClassificationDataset(X_ood_train,y_ood_train,transforms=transform)
                OOD_dataset_val = ClassificationDataset(X_ood_val,y_ood_val,transforms=transform)

            else: 
                X = [img for img, _ in ood_train_data]
                y = [label for _, label in ood_train_data]
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
            
                OOD_dataset_train = ClassificationDataset(X_train,y_train,transforms=transform)
                OOD_dataset_val = ClassificationDataset(X_val,y_val,transforms=transform)

            ood_test_loader = DataLoader(OOD_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_train_loader = DataLoader(OOD_dataset_train, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_val_loader = DataLoader(OOD_dataset_val, batch_size=batch_size, shuffle=False, num_workers=16)
    

    else:
        if not calculate_val:
            X_ood = [img for img, _ in ood_data]
            y_ood = [label for _, label in ood_data]

            X_ood_train = [img for img, _ in ood_train_data]
            y_ood_train = [label for _, label in ood_train_data]

            X_ood_val = [img for img, _ in ood_val_data]
            y_ood_val = [label for _, label in ood_val_data]

        
            OOD_dataset = ClassificationDataset(X_ood,y_ood,transforms=transform)
            OOD_dataset_train = ClassificationDataset(X_ood_train,y_ood_train,transforms=transform)
            OOD_dataset_val = ClassificationDataset(X_ood_val,y_ood_val,transforms=transform)

            ood_test_loader = DataLoader(OOD_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_train_loader = DataLoader(OOD_dataset_train, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_val_loader = DataLoader(OOD_dataset_val, batch_size=batch_size, shuffle=False, num_workers=16)
        
        else:
            X_ood = [img for img, _ in ood_data]
            y_ood = [label for _, label in ood_data]

            X = [img for img, _ in ood_train_data]
            y = [label for _, label in ood_train_data]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


            OOD_dataset = ClassificationDataset(X_ood,y_ood,transforms=transform)
            OOD_dataset_train = ClassificationDataset(X_train,y_train,transforms=transform)
            OOD_dataset_val = ClassificationDataset(X_val,y_val,transforms=transform)

            ood_test_loader = DataLoader(OOD_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_train_loader = DataLoader(OOD_dataset_train, batch_size=batch_size, shuffle=False, num_workers=16)
            ood_val_loader = DataLoader(OOD_dataset_val, batch_size=batch_size, shuffle=False, num_workers=16)

    if data_trainded == data_path:
        process_and_save(model , train_loader ,results_dir, f"id_train")
        process_and_save(model , val_loader ,results_dir, f"id_val")
        process_and_save(model , id_loader ,results_dir, f"id_test")
        if data_path == "super_cifar100":
            process_and_save(model , ood_test_loader ,results_dir, f"ood_test")
            process_and_save(model , ood_train_loader ,results_dir, f"ood_train")
            process_and_save(model , ood_val_loader ,results_dir, f"ood_val")
    else:    
        process_and_save(model , ood_test_loader ,results_dir, f"{data_path}_test")
        process_and_save(model , ood_train_loader ,results_dir, f"{data_path}_train")
        process_and_save(model , ood_val_loader ,results_dir, f"{data_path}_val")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to execute OOD detectors')
    parser.add_argument('-id', '--id-database', required=True, type=str, help='Database trained')
    parser.add_argument('-d', '--database', required=True, type=str, help='Database to process')
    parser.add_argument('-f','--fold', required=True, type=int, help='Fold to use')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output dir')
    parser.add_argument('-e', '--experiment-name', required=True, type=str, help='Experiment name')
    parser.add_argument('--epoch', required=False, default=None, type=str, help='Epoch')
    parser.add_argument('--seed', required=False, default=None, type=str, help='seed')
    parser.add_argument('-v', '--version', required=False, default=0, type=int, help='Version')
    parser.add_argument('--val', action=argparse.BooleanOptionalAction, help='Calculate validation to train')
    


    args = parser.parse_args()

    main(args)