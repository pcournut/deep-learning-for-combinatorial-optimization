#!/bin/bash
#PBS -N Google Active Search
#PBS -A esi-533-aa
#PBS -l walltime=01:00:00
#PBS -l nodes=1:gpus=2

module load compilers/gcc/4.8.5 cuda/7.5 libs/cuDNN/5
module load apps/python/3.5.0

source lizard/bin/activate

cd $HOME/Google_Pointer_Net
python main.py --nb_epoch 10000 --log_dir 'summary/10.Helios' > fichier.sortie