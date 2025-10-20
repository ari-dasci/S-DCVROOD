#!/bin/bash 

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"


seeds=("981127" "1291430" "540103" "723262" "1531415" "1846782" "620738" "326040" "1466620")


folds=("0" "1" "2" "3" "4")


iddatabases=("cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "super_cifar100" "super_cifar100")
ooddatabases=("cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "super_cifar100" "super_cifar100")
versions=("0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "1" "2" "3" "4" "5" "1" "2" "3" "4" "5" "0" "1")

postprocesoss=("ebo" "fdbd" "gen" "klm" "knn" "mds" "nnguide" "odin" "relation" "she")
experiments=("gt_vit_b_16")

conda activate novelty_detection
export PYTHONPATH='.'

for seed in "${seeds[@]}"; do
    for expemiment in "${experiments[@]}"; do
        for j in "${!iddatabases[@]}"; do
            id="${iddatabases[$j]}"
            ood="${ooddatabases[$j]}"
            version="${versions[$j]}"
            for postprocess in "${postprocesoss[@]}"; do
                for i in "${!folds[@]}"; do
                    fold="${folds[$i]}"
                    echo "Running: python src/exec_pp.py -id $id -ood $ood -f $fold -o results -p $postprocess -e $expemiment -v $version --seed $seed"
                    python src/exec_pp.py -id $id -ood $ood -f $fold -o results -p $postprocess -e $expemiment -v $version --seed $seed
                    echo "Run ended"
                done
            done
        done
    done
done