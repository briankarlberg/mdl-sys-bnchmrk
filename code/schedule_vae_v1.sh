#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=vae_v1
#SBATCH --time=9-00:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kirchgae@ohsu.edu;karlberb@ohsu.edu


epochs=$1
batch_size=$2
latent_space=$3

python3 ./vae_v1.py --epochs "${epochs}" --batch_size "${batch_size}" --latent_space "${latent_space}"
