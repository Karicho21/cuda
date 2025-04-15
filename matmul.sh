#!/bin/bash 
#SBATCH --job-name=MyGPUJob
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
module load cuda
nvidia-smi
nvcc -Xcompiler -fopenmp matmul.cu -lpthread -lcublas -o matmul

output="output.txt"

./matmul 512 4	>> $output
./matmul 1024 4	>> $output
./matmul 2048 4 >> $output



