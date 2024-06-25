#!/bin/bash
#SBATCH --job-name=gpgpu_practico_5_2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=federico.gil@fing.edu.uy
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

source /etc/profile.d/modules.sh

module load cuda/11.0

cd /clusteruy/home/gpgpu10

$1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15