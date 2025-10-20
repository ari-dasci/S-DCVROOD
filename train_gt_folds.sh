#!/bin/bash 

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate novelty_detection
export PYTHONPATH="."
numbers=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "52" "53" "54" "55" "56" "57" "58" "59" "60" "61" "62" "63" "64" "65" "66" "67" "68" "69" "70" "71" "72" "73" "74" "75" "76" "77" "78" "79" "80" "81" "82" "83" "84" "85" "86" "87" "88" "89" "90" "91" "92" "93" "94" "95" "96" "97" "98" "99")

iddatabases=("cifar10" "cifar100" "super_cifar100")

experiment="gt_vit_b_16"

for h in "${!iddatabases[@]}"; do
    
    id="${iddatabases[$h]}"

    for j in "${!numbers[@]}"; do
        fold="${numbers[$j]}"
        echo "Running: src/gt_trainer.py -e ${experiment}_${id}_fold_${fold} -o ./models/ -d ./gt_folds/${id} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight"
        python src/gt_trainer.py -e ${experiment}_${id}_fold_${fold} -o ./models/ -d ./gt_folds/${id} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight
        echo "Training completed"
    done
done

echo "All base trainings completed, starting outlier training"

iddatabases=("cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "super_cifar100")
ooddatabases=("cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "super_cifar100")
for h in "${!iddatabases[@]}"; do
    
    id="${iddatabases[$h]}"
    ood="${ooddatabases[$h]}"

    for j in "${!numbers[@]}"; do
        fold="${numbers[$j]}"
        echo "Running: src/gt_trainer.py -e ${experiment}_${id}_fold_${fold} -o ./models/ -d ./gt_folds/${id} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --version version_0 --epoch 01  --outliers ./gt_folds/${ood}"
        python src/gt_trainer.py -e ${experiment}_${id}_fold_${fold} -o ./models/ -d ./gt_folds/${id} -m vit_b_16 -f $fold -s 1 --train-batch-size 64 --val-batch-size 64 --enable-class-weight --version version_0 --epoch 01  --outliers ./gt_folds/${ood}
        echo "Training completed"
    done
done

echo "All outlier trainings completed"