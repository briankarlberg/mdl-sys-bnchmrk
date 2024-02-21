#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=vae_lsp_v1
#SBATCH --time=0-03:00:00
#SBATCH --partition=exacloud
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kirchgae@ohsu.edu,karlberb@ohsu.edu


epochs=$1
batch_size=$2
latent_space=$3
data=$4

python3 ./vae.py --epochs "${epochs}" --batch_size "${batch_size}" --latent_space "${latent_space}" --data "${data}" --output_dir "../results/vae/latent_spaces/v1"
