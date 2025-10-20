Hello dear reader.

The experimentation is made with the following code:

Download datasets:
 - download_datasets.py

Create folds:
 - create_folds.py
 - create_folds.superclasses.py 

Create ground truth folds:
 - create_gt_folds --seed $1 --number $2
    - Seed indicate a seed for reproducibility
    - number is a seed to create diferent folds

Train:
 - trainer.py
    This code has multiple parameters to train with diverse datasets, also it is posible to select a previous trained net and further training with outlier exposute

Embbedind extractors:
 - process_embs.py
    Extract the embeddings and logits of a trained net.

OOD detectors:
 - exec_pp.py
 - gt_exec_pp.py
    Given a trained net, an OOD detector, an ID dataset and an OOD dataset, these files calculate an OOD score.

 - exec_pp_embs.py
    Instead of a trained net you can use the embbedings and logits to calculate an OOD score (if possible).

Reading:
 - read_confidence.py
 - convergen.py
    Calculate metrics given an OOD score.