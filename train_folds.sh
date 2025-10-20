#!/bin/bash 

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate novelty_detection
export PYTHONPATH='.'
numbers=("0" "1" "2" "3" "4")
seeds=("981127" "1291430" "540103" "723262" "1531415" "1846782" "620738" "326040" "1466620")
iddatabases=("cifar10" "cifar100" "super_cifar100")

experiment="vit_b_16"

for seed in "${seeds[@]}"; do
    for h in "${!iddatabases[@]}"; do
        
        id="${iddatabases[$h]}"

        for j in "${!numbers[@]}"; do
            fold="${numbers[$j]}"
            echo "Running: src/gt_trainer.py -e ${experiment}_${id}_fold_${fold}_${seed} -o ./models/ -d ./folds/${id}_${seed} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --val"
            python src/gt_trainer.py -e ${experiment}_${id}_fold_${fold}_${seed} -o ./models/ -d ./folds/${id}_${seed} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --val
            echo "Training completed"
        done
    done
done

echo "All base trainings completed, starting outlier training"

iddatabases=("cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "super_cifar100")
ooddatabases=("cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "super_cifar100")
for seed in "${seeds[@]}"; do
    for h in "${!iddatabases[@]}"; do
        
        id="${iddatabases[$h]}"
        ood="${ooddatabases[$h]}"

        for j in "${!numbers[@]}"; do
            fold="${numbers[$j]}"
            echo "Running: src/gt_trainer.py -e ${experiment}_${id}_fold_${fold}_${seed} -o ./models/ -d ./folds/${id}_${seed} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --val  --version version_0 --epoch 01  --outliers ./folds/${ood}_${seed}"
            python src/gt_trainer.py -e ${experiment}_${id}_fold_${fold}_${seed} -o ./models/ -d ./folds/${id}_${seed} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --val  --version version_0 --epoch 01  --outliers ./folds/${ood}_${seed}
            echo "Training completed"
        done
    done
done

echo "All outlier trainings completed"