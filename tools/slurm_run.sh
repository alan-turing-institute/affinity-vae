#!/bin/bash -l
#SBATCH --account=vjgo8416-ms-img-pc # this is the account that will be charged for the job, change to your own account
#SBATCH --qos=turing # use the turing partition, change to your preferred partition
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --time 8:00:00 # maximum walltime
#SBATCH --cpus-per-gpu=36 # use 36 CPU cores per GPU
#SBATCH --ntasks-per-node=1 # use 1 task per node
#SBATCH --mem=0 # use all available memory on the node
#SBATCH --nodes=1 # use 1 node

# Load baskerville environment, created as described in main README
source /path/to/environment/pyenv_affinity/bin/activate

# Load the required modules
module purge
module load baskerville
module load bask-apps/live

module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load torchvision/0.15.2-foss-2022a-CUDA-11.7.0


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


srun python /path/to/affinity-vae/run.py --config_file path/to/avae-config_file --new_out
