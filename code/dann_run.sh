#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=dann
#SBATCH --time=0-03:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kirchgae@ohsu.edu,karlberb@ohsu.edu


data=$1
output_dir=$2

python3 ./DANN.py --data "${data}" --output_dir "${output_dir}" --latent_space 32
