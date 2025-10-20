import pandas as pd
import argparse
import sys
import os
import json
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, accuracy_score

experiments=["vit_b_16", "resnet18", "efficientnet_l"]
folds = [0,1,2,3,4]
postprocesoss=["ebo", "fdbd", "gen", "knn", "klm", "mds", "nnguide", "odin", "relation", "she"]
iddatabases=["cifar10", "cifar100", "cifar10", "cifar100", "super_cifar100"]
ooddatabases=["cifar100", "cifar10", "mnist", "mnist", "super_cifar100"]
versions=["version_0"]

# Function to find TPR@FPR threshold
def get_tpr_at_fpr(target_fpr, fpr, tpr):
    """Interpolate to find TPR at a fixed FPR level."""
    return np.interp(target_fpr, fpr, tpr)

# Function to find FPR@TPR threshold
def get_fpr_at_tpr(target_tpr, fpr, tpr):
    """Interpolate to find FPR at a fixed TPR level."""
    return np.interp(target_tpr, tpr, fpr)

def mejor_experimento(grupo):
    mejores = {
        "auroc": grupo.set_index("experiments")["auroc"].idxmax(),  
        "aupr": grupo.set_index("experiments")["aupr"].idxmax(),  
        "TPR@FPR": grupo.set_index("experiments")["TPR@FPR"].idxmax(),  
        "FPR@TPR": grupo.set_index("experiments")["FPR@TPR"].idxmin(),  
        "acc_99": grupo.set_index("experiments")["acc_99"].idxmax(),  
        "acc_95": grupo.set_index("experiments")["acc_95"].idxmax(),  
        "acc_90": grupo.set_index("experiments")["acc_90"].idxmax(),  
        "acc_80": grupo.set_index("experiments")["aupr"].idxmax()  

    }
    return pd.Series(mejores)

def main():

    summary_extended = dict()
    experiments_extended_list = []
    fold_extended_list = []
    post_extended_list = []
    id_extended_list = []
    ood_extended_list = []
    version_extended_list = []
    auroc_extended_list = []
    aupr_extended_list = []
    f1_extended_list = []
    tpr_at_fpr_extended_list = []
    fpr_at_tpr_extended_list = []
    accuracy_99_extended_list = []
    accuracy_95_extended_list = []
    accuracy_90_extended_list = []
    accuracy_80_extended_list = []

    summary = dict()
    experiments_list = []
    version_list = []
    post_list = []
    id_list = []
    ood_list = []
    auroc_mean_list = []
    aupr_mean_list = []
    f1_mean_list = []
    tpr_at_fpr_mean_list = []
    fpr_at_tpr_mean_list = []
    accuracy_99_mean_list = []
    accuracy_95_mean_list = []
    accuracy_90_mean_list = []
    accuracy_80_mean_list = []

    id_confidence = []
    ood_confidence = []

    mean_id_confidence = []
    mean_ood_confidence = []

    for experiment_name in experiments:
        for version in versions:
            for i in range(len(iddatabases)):
                for postprocessor_name in postprocesoss:
                    
                    lil_id = 0
                    lil_od = 0
                    mean_auroc = 0
                    mean_aupr = 0
                    mean_f1 = 0
                    mean_tpr_at_fpr = 0
                    mean_fpr_at_tpr = 0
                    mean_acc_99 = 0
                    mean_acc_95 = 0
                    mean_acc_90 = 0
                    mean_acc_80 = 0


                    experiments_list.append(experiment_name)
                    post_list.append(postprocessor_name)
                    id_list.append(iddatabases[i])
                    ood_list.append(ooddatabases[i])
                    version_list.append(version)

                    for fold in folds:
                        experiments_extended_list.append(experiment_name)
                        fold_extended_list.append(fold)
                        version_extended_list.append(version)
                        post_extended_list.append(postprocessor_name)
                        id_extended_list.append(iddatabases[i])
                        ood_extended_list.append(ooddatabases[i])

                        # Loading necessary data
                        output_path = os.path.join("results", f"fold_{fold}", postprocessor_name, f"{experiment_name}_{iddatabases[i]}_{ooddatabases[i]}_{version}")
                        
                        print(f"{fold}_{postprocessor_name}_{experiment_name}_{iddatabases[i]}_{ooddatabases[i]}" , flush=True)
                        with open(os.path.join(output_path, "threshold_2.json"), "r") as f:
                            threshold_dictionary = json.load(f)

                        threshold_dictionary = {int(k): v for k, v in threshold_dictionary.items()}
                        results = pd.read_csv(os.path.join(output_path, "results.csv"))

                        # CONFIDENCE
                        id_values_confidence = -results[results['Is_OOD'] == False]['Confidence'].to_numpy()
                        ood_values_confidence = -results[results['Is_OOD'] == True]['Confidence'].to_numpy()

                        id_confidence.append(id_values_confidence.mean())
                        ood_confidence.append(ood_values_confidence.mean())
                        lil_id = lil_id + id_values_confidence.sum()
                        lil_od = lil_od + ood_values_confidence.sum()
                        
                        # AUROC
                        auroc = roc_auc_score(- results['Is_OOD'], results['Confidence']) # Compute AUROC

                        auroc_extended_list.append(auroc)
                        mean_auroc = mean_auroc + auroc  

                        # AUPR
                        precision, recall, _ = precision_recall_curve(- results['Is_OOD'], results['Confidence'])
                        aupr = auc(recall, precision)  # Compute AUPR
                        aupr_extended_list.append(aupr)
                        mean_aupr = mean_aupr + aupr 

                        # F1 score

                        f1_scores = 2 * (precision * recall) / (precision + recall) # if precision + recall != 0 else 0
                        max_f1 = max(f1_scores)  # Maximum F1 score across all thresholds
                        f1_extended_list.append(max_f1)
                        mean_f1 = mean_f1 + max_f1 

                        # TPR@FPR (True Positive Rate at a fixed False Positive Rate) and FPR@TPR (False Positive Rate at a fixed True Positive Rate)
                        fpr, tpr, _ = roc_curve(- results['Is_OOD'], results['Confidence'])
                        tpr_at_fpr_5 = get_tpr_at_fpr(0.05, fpr, tpr)  # TPR@5% FPR
                        fpr_at_tpr_95 = get_fpr_at_tpr(0.95, fpr, tpr)  # FPR@95% TPR


                        tpr_at_fpr_extended_list.append(tpr_at_fpr_5)
                        fpr_at_tpr_extended_list.append(fpr_at_tpr_95)

                        mean_tpr_at_fpr = mean_tpr_at_fpr + tpr_at_fpr_5
                        mean_fpr_at_tpr = mean_fpr_at_tpr + fpr_at_tpr_95

                        # ACURACY
                        ## 99
                        pred_ood = (results['Confidence'] < threshold_dictionary[99]).astype(int).tolist()
                        accuracy_99 = accuracy_score(results['Is_OOD'], pred_ood)

                        accuracy_99_extended_list.append(accuracy_99)

                        mean_acc_99 = mean_acc_99 + accuracy_99

                        ## 95
                        pred_ood = (results['Confidence'] < threshold_dictionary[95]).astype(int).tolist()
                        accuracy_95 = accuracy_score(results['Is_OOD'], pred_ood)

                        accuracy_95_extended_list.append(accuracy_95)

                        mean_acc_95 = mean_acc_95 + accuracy_95

                        ## 90
                        pred_ood = (results['Confidence'] < threshold_dictionary[90]).astype(int).tolist()
                        accuracy_90 = accuracy_score(results['Is_OOD'], pred_ood)

                        accuracy_90_extended_list.append(accuracy_90)

                        mean_acc_90 = mean_acc_90 + accuracy_90

                        ## 80
                        pred_ood = (results['Confidence'] < threshold_dictionary[80]).astype(int).tolist()
                        accuracy_80 = accuracy_score(results['Is_OOD'], pred_ood)

                        accuracy_80_extended_list.append(accuracy_80)

                        mean_acc_80 = mean_acc_80 + accuracy_80

                        
                    # CONFIDENCE
                    mean_id_confidence.append(lil_id / len(folds))
                    mean_ood_confidence.append(lil_od / len(folds))    

                    # AUROC
                    mean_auroc = mean_auroc / len(folds)
                    auroc_mean_list.append(mean_auroc)

                    # AUPR
                    mean_aupr = mean_aupr / len(folds)
                    aupr_mean_list.append(mean_aupr)

                    # F1
                    mean_f1 = mean_f1 / len(folds)
                    f1_mean_list.append(mean_f1)

                    # TPR@FPR (True Positive Rate at a fixed False Positive Rate) and FPR@TPR (False Positive Rate at a fixed True Positive Rate)
                    mean_tpr_at_fpr = mean_tpr_at_fpr / len(folds)
                    tpr_at_fpr_mean_list.append(mean_tpr_at_fpr)

                    mean_fpr_at_tpr = mean_fpr_at_tpr / len(folds)
                    fpr_at_tpr_mean_list.append(mean_fpr_at_tpr)

                    # ACCURACY
                    ## 99
                    mean_acc_99 = mean_acc_99 / len(folds)
                    accuracy_99_mean_list.append(mean_acc_99)

                    ## 95
                    mean_acc_95 = mean_acc_95 / len(folds)
                    accuracy_95_mean_list.append(mean_acc_95)

                    ## 90
                    mean_acc_90 = mean_acc_90 / len(folds)
                    accuracy_90_mean_list.append(mean_acc_90)

                    ## 80
                    mean_acc_80 = mean_acc_80 / len(folds)
                    accuracy_80_mean_list.append(mean_acc_80)
                
    
    summary_extended['experiments'] = experiments_extended_list
    summary_extended['versions'] = version_extended_list
    summary_extended['postprocess'] = post_extended_list
    summary_extended['id_database'] = id_extended_list
    summary_extended['ood_database'] = ood_extended_list
    summary_extended['fold'] = fold_extended_list
    summary_extended['id_conf'] = id_confidence
    summary_extended['ood_conf'] = ood_confidence
    summary_extended['auroc'] = auroc_extended_list
    summary_extended['aupr'] = aupr_extended_list
    summary_extended['F1_score'] = f1_extended_list
    summary_extended['TPR@FPR'] = tpr_at_fpr_extended_list
    summary_extended['FPR@TPR'] = fpr_at_tpr_extended_list
    summary_extended['acc_99'] = accuracy_99_extended_list
    summary_extended['acc_95'] = accuracy_95_extended_list
    summary_extended['acc_90'] = accuracy_90_extended_list
    summary_extended['acc_80'] = accuracy_80_extended_list

    sumary_extended_pd = pd.DataFrame(summary_extended)

    summary['experiments'] = experiments_list
    summary['versions'] = version_list
    summary['postprocess'] = post_list
    summary['id_database'] = id_list
    summary['ood_database'] = ood_list
    summary['id_conf'] = mean_id_confidence
    summary['ood_conf'] = mean_ood_confidence
    summary['auroc'] = auroc_mean_list
    summary['aupr'] = aupr_mean_list
    summary['F1_score'] = f1_mean_list
    summary['TPR@FPR'] = tpr_at_fpr_mean_list
    summary['FPR@TPR'] = fpr_at_tpr_mean_list
    summary['acc_99'] = accuracy_99_mean_list
    summary['acc_95'] = accuracy_95_mean_list
    summary['acc_90'] = accuracy_90_mean_list
    summary['acc_80'] = accuracy_80_mean_list

    sumary_pd = pd.DataFrame(summary)
    
    sumary_extended_pd.to_excel("results/summary_extended.xlsx", index=False)
    sumary_pd.to_excel("results/summary.xlsx", index=False)

    sumary_extended_pd.to_csv("results/summary_extended.csv", index=False)
    sumary_pd.to_csv("results/summary.csv", index=False)


    

if __name__ == '__main__':


    main()