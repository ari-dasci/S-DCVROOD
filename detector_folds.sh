#!/bin/bash 

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"


seeds=("981127" "1291430" "540103" "723262" "1531415" "1846782" "620738" "326040" "1466620")


folds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "52" "53" "54" "55" "56" "57" "58" "59" "60" "61" "62" "63" "64" "65" "66" "67" "68" "69" "70" "71" "72" "73" "74" "75" "76" "77" "78" "79" "80" "81" "82" "83" "84" "85" "86" "87" "88" "89" "90" "91" "92" "93" "94" "95" "96" "97" "98" "99")


iddatabases=("cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "cifar10" "cifar10" "cifar10" "cifar10" "cifar10" "cifar100" "cifar100" "cifar100" "cifar100" "cifar100" "super_cifar100" "super_cifar100")
ooddatabases=("cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar100" "mnist" "letters" "dtd" "tiny_imagenet_200" "cifar10" "mnist" "letters" "dtd" "tiny_imagenet_200" "super_cifar100" "super_cifar100")
versions=("0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "1" "2" "3" "4" "5" "1" "2" "3" "4" "5" "0" "1")

postprocesoss=("ebo" "fdbd" "gen" "klm" "knn" "mds" "nnguide" "odin" "relation" "she")
experiments=("vit_b_16")

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