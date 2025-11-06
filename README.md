# S-DCVROOD
Software, experiments and results of the paper [DCV-ROOD Evaluation Framework: Dual Cross-Validation for Robust Out-of-Distribution Detection](https://arxiv.org/abs/2509.05778)

## Usage

Download datasets:
 - download_datasets.py
     Some version cannot download emnist with torch, in that case download in a different way.

Create folds:
 - create_folds.py
 - create_folds.superclasses.py 

Create ground truth folds:
 - create_gt_folds --seed $1 --number $2
    - Seed indicate a seed for reproducibility
    - number is a seed to create diferent folds

Train:
 - trainer.py
    This code has multiple parameters to train with diverse datasets, also it is posible to select a previous trained net and further training with outlier exposure.

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

## Full experiments and results

If you are interested in reviewing the complete results of the experiments conducted in the paper, you can access them through this [link]().



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{urrea2025dcv,
  title={DCV-ROOD Evaluation Framework: Dual Cross-Validation for Robust Out-of-Distribution Detection},
  author={Urrea-Casta{\~n}o, Arantxa and Segura-Kunsagi, Nicol{\'a}s and Su{\'a}rez-D{\'i}az, Juan Luis and Montes, Rosana and Herrera, Francisco},
  journal={arXiv preprint arXiv:2509.05778},
  year={2025}
}
```

(Pre-print citation, to be updated upon publication.)
