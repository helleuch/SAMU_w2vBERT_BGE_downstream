#!/bin/bash

#SBATCH --job-name=SLU_SM_EP1_SLOTS_ASR
#SBATCH --account=slp@a100
#SBATCH -C a100
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --output=log/SLU_SM_EP1_SLOTS_ASR-%j.log
#SBATCH --ntasks-per-node=1     # we will run once the next srun per node
#SBATCH --gres=gpu:1            # we want N GPUs per node


module purge

module load sox/14.4.2
module load anaconda-py3/2023.09
module load git/2.39.1

cd /lustre/fswork/projects/rech/slp/uxx18ce/SAMU_w2vBERT_BGE_downstream/speech_MASSIVE

conda activate a100

nvidia-smi

srun python train.py hparams/train_SAMU_1_epoch_alignment.yaml

echo "Job done."
