#!/bin/bash
#SBATCH -J NewTest_VERBNOUN_seq25
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
cd $HOME/Thesis/Test
module load python/3.8.6
python main3Test2.py -remark 'NewTest_VERBNOUN_seq25_'\
		       -seqlen 25 -embeddingDim 256 -hiddenLayer 3 -hiddenSize 256 -bidirection 1 -withHiddenState 1 \
                       -numconv1d 0 -groupRelu 1 -convpredict 0 -predictkernelsize 2 \
                       -interventionP 1 -minmerge 2 -maxmerge 3 -mergeRate 3 \
                       -biasTrain 0 -trainSize 15000 -maxlen 600 -batch 1000 -slide 25 -optstep 1 -dynamicOpt 0 -epoch 1000 \
		       -preprocess 1 -onlyMerge VERB NOUN -consecutive 1
