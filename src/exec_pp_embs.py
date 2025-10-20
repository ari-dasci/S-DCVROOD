from types import SimpleNamespace

import torch
from torchvision import transforms

import pandas as pd
from tqdm.auto import tqdm
import argparse
import sys
import os
import numpy as np
import json

from trainer import ClassificationModel
from src.dataset_loader import ClassificationDataset

import yaml

import src.postprocessors as pp

postporcessor_class = {
    'base': lambda config: pp.BasePostprocessor(config),
    'dice': lambda config: pp.DICEPostprocessor(config),
    'ebo': lambda config: pp.EBOPostprocessor(config),
    'fdbd': lambda config: pp.fDBDPostprocessor(config),
    'gen': lambda config: pp.GENPostprocessor(config),
    'klm': lambda config: pp.KLMatchingPostprocessor(config),
    'knn': lambda config: pp.KNNPostprocessor(config),
    'mds': lambda config: pp.MDSPostprocessor(config),
    'nnguide': lambda config: pp.NNGuidePostprocessor(config),
    'odin': lambda config: pp.ODINPostprocessor(config),
    'relation': lambda config: pp.RelationPostprocessor(config),
    'she': lambda config: pp.SHEPostprocessor(config) 
}


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))  # Convierte subdiccionarios en objetos
            else:
                setattr(self, key, value)

def main(args):

    id_database = args.id_data
    ood_database = args.ood_data
    experiment_name = args.experiment_name
    output_dir = args.output_dir
    fold = args.fold
    epoch = args.epoch
    version = args.version
    postprocessor_name = args.postprocessor_name
    only_threshold = args.only_threshold

    batch_size = 64
    torch.set_float32_matmul_precision('medium')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_dir = os.path.join(output_dir, f"fold_{fold}", postprocessor_name, f"{experiment_name}_{id_database}_{ood_database}_version_{version}")

    if os.path.exists(f"{results_dir}/results.csv") and not only_threshold:
        print(f'Features already generated at {results_dir}/results.csv. Delete it to regenerate the features')
        return 0

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 64

    if "mnist" in id_database or "letters" in id_database:
        transform_id = transforms.Compose([transforms.Grayscale(num_output_channels=3), transform])
    else:
        transform_id = transform

    if "mnist" in ood_database or "letters" in ood_database:
        transform_ood = transforms.Compose([ transforms.Grayscale(num_output_channels=3), transform])
    else:
        transform_ood = transform

    if experiment_name == "resnet18":
        transform_id = transforms.Compose([transform_id, transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])]) 
        transform_ood = transforms.Compose([transform_ood, transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
    elif experiment_name == "efficientnet_l":
        batch_size = 32
        transform_id = transforms.Compose([transform_id, transforms.Resize((480, 480)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        transform_ood = transforms.Compose([transform_ood, transforms.Resize((480, 480)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    elif experiment_name == "vit_b_16" or experiment_name == "gt_vit_b_16":
        batch_size = 64
        transform_id = transforms.Compose([transform_id, transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_ood = transforms.Compose([transform_ood, transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        print("ERROR UNRECOGNISED MODEL")
        sys.exit(-1)


    # Load model
    # Load model
    if epoch!=None:
        model_path = os.path.join("models", f"{experiment_name}_{id_database}_fold_{fold}", f"version_{version}", "checkpoints", f"{experiment_name}_{id_database}_fold_{fold}_epoch={epoch}-val_loss=0.00.ckpt")
    else:
        checkpoints_dir = os.path.join("models", f"{experiment_name}_{id_database}_fold_{fold}", f"version_{version}", "checkpoints")
        checkpoints_names = list(os.listdir(checkpoints_dir))
        model_path = os.path.join(checkpoints_dir, checkpoints_names[-1])

    model = ClassificationModel.load_from_checkpoint(model_path)

    print("Model loaded",flush=True)
    model.to(device)
    model.eval()

    with open(f"./src/configs/{postprocessor_name}.yml") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            
        except yaml.YAMLError as exc:
            print(exc)
    
    aux = dict()
    aux['name'] = id_database
    config_dict['dataset'] = aux

    config = DictToObj(config_dict)
    postprocessor = postporcessor_class[postprocessor_name](config=config)
    print("Postprocessor loaded",flush=True)

    id_path = os.path.join("embs", f"{experiment_name}_{id_database}_version_{version}_fold_{fold}")
    ood_path = os.path.join("embs", f"{experiment_name}_{id_database}_version_{version}_fold_{fold}")


    id_train = pd.read_csv(os.path.join(id_path, "id_train.csv"))
    id_val = pd.read_csv(os.path.join(id_path, "id_val.csv"))
    id_test = pd.read_csv(os.path.join(id_path, "id_test.csv"))

    if id_database != ood_database:
        ood_test = pd.read_csv(os.path.join(ood_path, f"{ood_database}_test.csv"))
    else:
        ood_test = pd.read_csv(os.path.join(ood_path, f"ood_test.csv"))

    

    print("Data Loaded",flush=True)

    id_dict = dict()
    id_dict["train"] = dict()
    id_dict["val"] = dict()

    id_dict["train"]["embeddings"] = torch.from_numpy(id_train.filter(regex='^embeddings').to_numpy())
    id_dict["train"]["logits"] = torch.from_numpy(id_train.filter(regex='^logits').to_numpy())
    id_dict["train"]["labels"] = torch.from_numpy(id_train.filter(regex='^labels').to_numpy())

    id_dict["val"]["embeddings"] = torch.from_numpy(id_val.filter(regex='^embeddings').to_numpy())
    id_dict["val"]["logits"] = torch.from_numpy(id_val.filter(regex='^logits').to_numpy())
    id_dict["val"]["labels"] = torch.from_numpy(id_val.filter(regex='^labels').to_numpy())

    validating_id = dict()
    validating_id["embeddings"] = torch.from_numpy(id_val.filter(regex='^embeddings').to_numpy())
    validating_id["logits"] = torch.from_numpy(id_val.filter(regex='^logits').to_numpy())

    testing_id = dict()
    testing_id["embeddings"] = torch.from_numpy(id_test.filter(regex='^embeddings').to_numpy())
    testing_id["logits"] = torch.from_numpy(id_test.filter(regex='^logits').to_numpy())


    testing_ood = dict()
    testing_ood["embeddings"] = torch.from_numpy(ood_test.filter(regex='^embeddings').to_numpy())
    testing_ood["logits"] = torch.from_numpy(ood_test.filter(regex='^logits').to_numpy())

    

    print("Data prepared", flush=True)
    postprocessor.setup_emb(model, id_dict, None)
    print("Postprocessor set up", flush=True)
    
    thresshold_data = []

    print("Calculating threshold")
    _, confidence = postprocessor.postprocess_emb(model, validating_id)
    thresshold_data =  thresshold_data + confidence.cpu().numpy().tolist()

    thresshold_data = np.array(thresshold_data)
    threshold_dict = dict()

    threshold_dict[99] = np.percentile(thresshold_data, 1)
    threshold_dict[95] = np.percentile(thresshold_data, 5)
    threshold_dict[90] = np.percentile(thresshold_data, 10)
    threshold_dict[80] = np.percentile(thresshold_data, 20)

    if not only_threshold:
        pred_list = []
        confidence_list = []
        is_ood_gr_list = []    

        print("Processing ID test", flush=True)
        pred, confidence = postprocessor.postprocess_emb(model, testing_id)
        pred_list = pred_list + pred.cpu().numpy().tolist()
        confidence_list =  confidence_list + confidence.cpu().numpy().tolist()
        is_ood_gr_list = is_ood_gr_list + [False]*len(testing_id["logits"])
            
            

        print("Processing OOD test", flush=True)
        pred, confidence = postprocessor.postprocess_emb(model, testing_ood)
        pred_list = pred_list + pred.cpu().numpy().tolist()
        confidence_list =  confidence_list + confidence.cpu().numpy().tolist()
        is_ood_gr_list = is_ood_gr_list + [True]*len(testing_ood["logits"])

        print("Processing ended", flush=True)

    print("Saving...")
    if not only_threshold:
        results = {
            'Prediction': pred_list,
            'Confidence': confidence_list,
            'Is_OOD': is_ood_gr_list
        }
        # Create a pandas DataFrame
        df = pd.DataFrame(results)

        # Save DataFrame to CSV
        df.to_csv(f"{results_dir}/results.csv", index=False)
        print("Results saved to results.csv")

    

    with open(f"{results_dir}/threshold.json", "w") as f:
        json.dump(threshold_dict, f)

    print("Thresholds saved to threshold.json")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to execute OOD detectors')
    parser.add_argument('-id', '--id-data', required=True, type=str, help='ID database')
    parser.add_argument('-ood', '--ood-data', required=True, type=str, help='OOD database')
    parser.add_argument('-f','--fold', required=True, type=int, help='Fold to use')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output dir')
    parser.add_argument('-e', '--experiment-name', required=True, type=str, help='Experiment name')
    parser.add_argument('--epoch', required=False, default=None, type=str, help='Epoch')
    parser.add_argument('-v', '--version', required=False, default=0, type=int, help='Version')
    parser.add_argument('-p', '--postprocessor-name', required=True, type=str, help='Postprocessor to use')
    parser.add_argument('--only-threshold', action=argparse.BooleanOptionalAction, help='If enabled, only the treshold is calculated')



    args = parser.parse_args()

    main(args)