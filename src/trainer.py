import numpy as np
from sklearn.utils import compute_class_weight
import argparse
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.optim import Adam


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.checkpoint import checkpoint # Chekpoint forward propagation

import torch.nn.functional as F

from dataset_loader import OutlinerDataset

class ClassificationModel(pl.LightningModule):
    def __init__(self, base_model, num_classes, hyperparameters, classes_weights=None, method=None):
        super(ClassificationModel, self).__init__()
        # Default hyperparameters
        self._hyperparameters = hyperparameters
        self._num_classes = num_classes
        self._base_model = base_model

        if self._base_model == "resnet18":

            self._model = models.resnet18(pretrained=True)

            if not hasattr(self._model, "fc") or not isinstance(self._model.fc, nn.Sequential):
                in_features = self._model.fc.in_features
                self._model.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
                )
        
        elif self._base_model == "efficientnet_v2_l":
            # Load backbone
            self._model = torch.hub.load("pytorch/vision:v0.16.2", "efficientnet_v2_l" , weights='IMAGENET1K_V1')

            # Set the new layers
            in_features = self._model.features[-1].out_channels

            self._model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
            )
        
        elif self._base_model == "vit_b_16":
            # Load backbone
            self._model = torch.hub.load("pytorch/vision:v0.16.2", "vit_b_16", weights='IMAGENET1K_V1')

            # Set the new layers
            in_features = self._model.heads[0].in_features

            self._model.heads = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
            )

        # # Freeze all layers first
        # for param in self._model.parameters():
        #      param.requires_grad = False

        # # Unfreeze the last 3 layers
        # for param in list(self._model.parameters())[-3:]:  
        #     param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=self._hyperparameters['embedding_size'], out_features=num_classes, bias=True)
        )

        self.classes_weights = classes_weights

        self.criterion = nn.CrossEntropyLoss(torch.from_numpy(self.classes_weights).float())

        # Metrics
        self._val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self._val_loss = MeanMetric()
        self._val_outlier_loss = MeanMetric()
        
        if method is not None:
            self.method = method

        # Store hyperparameters
        self.save_hyperparameters()
        

    def energy_score(self, logits):
        """Compute Energy Score for EnergyOE."""
        return -1.0* torch.logsumexp(logits / 1.0, dim=1)

    def training_step(self, batch, batch_idx, dataloader_idx=0):

       # Retrieve data
        images, labels, outliers = batch
        loss = 0
        
        # Compute predicted labels
        logits = self(images)

        inlier_mask = ~outliers  # Inliers (normal data)
        outlier_mask = outliers   # Outliers (OOD data)

        if inlier_mask.any():  # If there are inliers in the batch
            inlier_logits = logits[inlier_mask]
            inlier_labels = labels[inlier_mask]

            classification_loss = self.criterion(inlier_logits, inlier_labels)
            self.log("train/classification_loss", classification_loss)
            loss += classification_loss  # Add classification loss

            self.log("train/classification_loss", loss)
            
        if outlier_mask.any():  # If there are outliers in the batch
            outlier_logits = logits[outlier_mask]
            
            if self.method == "softmax":  # Basic OE (Uniform KL-Div Loss)
                uniform_target = torch.ones_like(outlier_logits) / outlier_logits.size(1)
                outlier_loss = F.kl_div(F.log_softmax(outlier_logits, dim=1), uniform_target, reduction="batchmean")
            
            elif self.method == "energy":  # EnergyOE (Softplus on Energy Score)
                energy = self.energy_score(outlier_logits)
                outlier_loss = torch.mean(F.softplus(energy))
            
            else:
                raise ValueError("Invalid method! Choose 'softmax' or 'energy'.")
            
            self.log("train/outlier_loss", outlier_loss)
            loss += outlier_loss  # Add outlier loss


        return loss


    def validation_step(self, batch, batch_idx):

        images, labels, outliers = batch
        
        # Compute predicted labels
        logits = self(images)

        inlier_mask = ~outliers  # Inliers (normal data)
        outlier_mask = outliers   # Outliers (OOD data)

        if inlier_mask.any():  # If there are inliers in the batch
            inlier_logits = logits[inlier_mask]
            inlier_labels = labels[inlier_mask]

            loss = self.criterion(inlier_logits, inlier_labels)
            labels_ids_preds = torch.argmax(inlier_logits, dim=1)

            # Compute metrics
            self._val_accuracy.update(labels_ids_preds, inlier_labels)
            self._val_loss.update(loss)

            # Log metrics
            self.log_dict({'val/class_loss': self._val_loss, 'val/accuracy': self._val_accuracy}, on_step=False, on_epoch=True)
            
        if outlier_mask.any():  # If there are outliers in the batch
            outlier_logits = logits[outlier_mask]
            
            if self.method == "softmax":  # Basic OE (Uniform KL-Div Loss)
                uniform_target = torch.ones_like(outlier_logits) / outlier_logits.size(1)
                loss = F.kl_div(F.log_softmax(outlier_logits, dim=1), uniform_target, reduction="batchmean")
            elif self.method == "energy":  # EnergyOE (Softplus on Energy Score)
                energy = self.energy_score(outlier_logits)
                loss = torch.mean(F.softplus(energy))

            self._val_outlier_loss.update(loss)

        
        self.log_dict({'val/class_loss': self._val_loss, 'val/outlier_loss': self._val_outlier_loss, 'val/accuracy': self._val_accuracy}, on_step=False, on_epoch=True)



    def forward(self, images, return_feature=False):
        if self._base_model == "efficientnet_v2_l":
            x = images

            # Apply gradient to the layers due to memory restrictions
            for layer in self._model.features: 
                x = checkpoint(layer, x, use_reentrant=False)

            x = torch.mean(x, dim=[2, 3]).requires_grad_() # CAREFULL can do a sneaky detachment

            for layer in self._model.classifier: 
                x = checkpoint(layer, x, use_reentrant=False)

            embeddings = torch.clone(x)

            for layer in self.classifier: 
                x = checkpoint(layer, x, use_reentrant=False)
            logits = torch.clone(x)
        
        else:
            embeddings = self._model(images)
            logits = self.classifier(embeddings)
        
        if return_feature:
            return logits.float(), embeddings.float()
        else:
            return logits.float()

    def configure_optimizers(self):
        for param in self.parameters():
            param.requires_grad = True
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self._hyperparameters['lr'], 
            betas=self._hyperparameters['betas'], 
            eps=self._hyperparameters['eps'], 
            weight_decay=self._hyperparameters['weight_decay'])
        return optimizer



    def get_fc(self):
        return self.classifier[-1]

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

    torch.set_float32_matmul_precision('medium') 
    torch.manual_seed(2809)

    # Configuration
    hyperparameters = {
        'batch_size': 64,
        'val_batch_size': 64,
        'lr': 0.0001,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0,
        'num_epochs': 30,
        'val_check_interval': 0.5,
        'patience': 3,
        'embedding_size': 256
    }
    data_path = args.data_path
    output_base_dir = args.output_base_dir
    experiment_name = args.experiment_name
    enable_class_weight = args.enable_class_weight
    version = args.version
    hyperparameters['batch_size'] = args.train_batch_size
    hyperparameters['val_batch_size']  = args.val_batch_size
    save_epoch = args.save_epoch
    fold = args.fold_split
    base_model = args.model
    apply_ood = args.ood
    calculate_val = args.val
    outliers = args.outliers
    epoch_load = args.epoch

    # Get last checkpoint path
    last_checkpoint_path = None
    name_of_file = experiment_name
    if version is not None:
        checkpoints_dir = os.path.join(output_base_dir, experiment_name, version, 'checkpoints')
        checkpoints_names = list(os.listdir(checkpoints_dir))
        matching_epoch = [s for s in checkpoints_names if f"epoch={epoch_load}" in s]
        last_checkpoint_path = os.path.join(checkpoints_dir, matching_epoch[0])

    # Load the fold
    if not calculate_val:
        if not apply_ood:
            train_data = torch.load(f"{data_path}/train_strat_fold_{fold}.pt")
            val_data = torch.load(f"{data_path}/val_strat_fold_{fold}.pt")
        else:  
            train_data = torch.load(f"{data_path}/train_ID_fold_{fold}.pt")
            val_data = torch.load(f"{data_path}/val_ID_fold_{fold}.pt")

        if outliers is not None:
            if not apply_ood:
                outliers_train_data = torch.load(f"{outliers}/train_group_fold_{fold}.pt")
                outliers_val_data = torch.load(f"{outliers}/val_group_fold_{fold}.pt")
            else:  
                outliers_train_data = torch.load(f"{outliers}/train_OOD_fold_{fold}.pt")
                outliers_val_data = torch.load(f"{outliers}/val_OOD_fold_{fold}.pt")
    
    else: 
        # Load the fold
        if not apply_ood:
            train_data = torch.load(f"{data_path}/train_strat_fold_{fold}.pt")
        else:  
            train_data = torch.load(f"{data_path}/train_ID_fold_{fold}.pt")

        if outliers is not None:
            if not apply_ood:
                outliers_train_data = torch.load(f"{outliers}/train_group_fold_{fold}.pt")
            else:  
                outliers_train_data = torch.load(f"{outliers}/train_OOD_fold_{fold}.pt")

    # Define transformations (Normalization for CIFAR-10 / CIFAR100) 
    if base_model == "resnet18":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
    elif base_model == "efficientnet_v2_l":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((480, 480)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    elif base_model == "vit_b_16":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        print("ERROR UNRECOGNISED MODEL")
        sys.exit(-1)

    if outliers is not None:
        if "mnist" in outliers or "letters" in outliers:
            transform_outlier = transforms.Compose([ transforms.Grayscale(num_output_channels=3), transform])
        else:
            transform_outlier = transform

    # Convert to tensors
    if not calculate_val:
        X_train = [img for img, _ in train_data]
        X_val = [img for img, _ in val_data]
        y_train = [label for _, label in train_data]
        y_val = [label for _, label in val_data]

        if apply_ood:
            auxliary = np.arange(0, len(np.unique(y_train)))
            maping = {key: swap for key,swap in zip(np.unique(y_train),auxliary)}
            y_train = [maping[i] for i in y_train]
            y_val = [maping[i] for i in y_val]
        
        
        if outliers is not None:
            X_train_outlier = [img for img, _ in outliers_train_data]
            X_val_outlier = [img for img, _ in outliers_val_data]
            y_train_outlier = [label for _, label in outliers_train_data]
            y_val_outlier = [label for _, label in outliers_val_data]
            if not apply_ood:
                num_outliers = int(0.1 * len(X_train))
                if len(X_train_outlier) < num_outliers:
                    num_outliers = len(X_train_outlier) - 100
                sss = StratifiedShuffleSplit(n_splits=1, train_size=num_outliers, random_state=42)
                for train_idx, _ in sss.split(X_train_outlier,y_train_outlier):
                    X_train_outlier = [X_train_outlier[i] for i in train_idx]
                    y_train_outlier = [y_train_outlier[i] for i in train_idx]

                num_outliers = int(0.1 * len(X_val))
                if len(X_val_outlier) < num_outliers:
                    num_outliers = len(X_val_outlier) - 50
                sss = StratifiedShuffleSplit(n_splits=1, train_size=num_outliers, random_state=42)
                for train_idx, _ in sss.split(X_val_outlier,y_val_outlier):
                    X_val_outlier = [X_val_outlier[i] for i in train_idx]
                    y_val_outlier = [y_val_outlier[i] for i in train_idx]
                    
            train_dataset = OutlinerDataset(X_train,y_train, X_train_outlier,y_train_outlier,transforms=transform, transforms_2 = transform_outlier)
            val_dataset = OutlinerDataset(X_val,y_val, X_val_outlier,y_val_outlier,transforms=transform, transforms_2 = transform_outlier)
        
        else:
            train_dataset = OutlinerDataset(X_train,y_train,transforms=transform)
            val_dataset = OutlinerDataset(X_val,y_val,transforms=transform)

        

        train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, num_workers=15, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=hyperparameters["val_batch_size"], shuffle=False, num_workers=15, pin_memory=False)
    
    else:
        X = [img for img, _ in train_data]
        y = [label for _, label in train_data]
        if apply_ood:
            auxliary = np.arange(0, len(np.unique(y)))
            maping = {key: swap for key,swap in zip(np.unique(y),auxliary)}
            y = [maping[i] for i in y]

        
        if outliers is not None:
            X_outlier = [img for img, _ in outliers_train_data]
            y_outlier = [label for _, label in outliers_train_data]
            num_outliers = int(0.1 * len(X))
            if len(X_outlier) < num_outliers:
                num_outliers = len(X_outlier) - 100
            sss = StratifiedShuffleSplit(n_splits=1, train_size=num_outliers, random_state=42)
            for train_idx, _ in sss.split(X_outlier,y_outlier):
                X_outlier = [X_outlier[i] for i in train_idx]
                y_outlier = [y_outlier[i] for i in train_idx]
            train_loader,val_loader,y_train = obtain_data(X,y,hyperparameters, transform, X_outlier,y_outlier, transform_2 = transform_outlier)

        else:
            train_loader,val_loader, y_train = obtain_data(X,y,hyperparameters, transform)

    if enable_class_weight:
        class_weight = compute_class_weight("balanced", classes= np.unique(np.array(y_train)), y=np.array(y_train))
    else:
        class_weight = None

    if outliers is not None:
        model = ClassificationModel(base_model, len(np.unique(y_train)), hyperparameters, class_weight , "softmax")
    else:
        model = ClassificationModel(base_model, len(np.unique(y_train)), hyperparameters, class_weight)

    

    # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    model.train()

    # Training loop
    a = ModelCheckpoint(filename=name_of_file+'_{epoch:02d}-{val_loss:.2f}', monitor='val/accuracy',
                        save_top_k=-1, # Save all checkpoints
                        every_n_epochs=save_epoch, # Save checkpoint every n epochs
                        save_on_train_epoch_end = True # Save at then end of the Epoch
                        )

    callbacks = [a]
    if hyperparameters['patience'] is not None:
        early_stop_callback = EarlyStopping(monitor="val/accuracy", min_delta=0.00, patience=hyperparameters['patience'], verbose=False, mode="max")
        callbacks.append(early_stop_callback)

    logger = TensorBoardLogger(output_base_dir, name=experiment_name)

    if outliers is not None:

        trainer = pl.Trainer(
            devices=[0], # Select GPU/cuda device if using cuda
            accelerator="gpu", # Indicate to use GPU
            logger=logger, 
            accumulate_grad_batches=1,
            default_root_dir=os.path.join(output_base_dir, experiment_name),
            callbacks=callbacks,
            val_check_interval=hyperparameters['val_check_interval'],
            min_epochs=7,
            max_epochs=8)
    
    else:
        trainer = pl.Trainer(
            devices=[1],
            accelerator="gpu", 
            logger=logger, 
            max_epochs=hyperparameters['num_epochs'], 
            accumulate_grad_batches=1,
            default_root_dir=os.path.join(output_base_dir, experiment_name),
            callbacks=callbacks,
            val_check_interval=hyperparameters['val_check_interval'])
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train our models')
    parser.add_argument('-d', '--data-path', required=True, type=str, help='Root data path')
    parser.add_argument('-o', '--output-base-dir', type=str, required=True, help='Directory where all models are stored')
    parser.add_argument('-e', '--experiment-name', type=str, required=True, help='Name of the experiment used to reference it in the future')
    parser.add_argument('--train-batch-size', type=int, required=True, help='Batch size in training phase')
    parser.add_argument('--val-batch-size', type=int, required=True, help='Batch size in inference phase')
    parser.add_argument('-v', '--version', default=None, help='If provided, the training will continue from that training version')
    parser.add_argument('-s', '--save-epoch', type=int, default=5, help='Save the model in the current epoch each X epochs')
    parser.add_argument('-f','--fold-split', type=str, required=True, help='Fold split used')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model to use')
    parser.add_argument('--ood', action=argparse.BooleanOptionalAction, help='OOD fold trainig')
    parser.add_argument('--val', action=argparse.BooleanOptionalAction, help='Calculate validation to train')
    parser.add_argument('--enable-class-weight', action=argparse.BooleanOptionalAction, help='If enabled, loss function will be weighted according to the train classes distribution')
    parser.add_argument('--outliers', type=str, required=False, default=None, help='Indicate from where take the outlier data')
    parser.add_argument('-ep','--epoch', default=None, help='If provided, the training will continue from that epoch in training')

    args = parser.parse_args()

    main(args)