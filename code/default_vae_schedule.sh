#!/bin/bash
directory=$1
output_directory=$2

# iterate through directory and select all .tsv files
for file in $directory/*.tsv; do
    echo "Running default_vae.py on ${file} with output directory ${output_directory}"
    # run the default_Vae.py script
    sbatch ./default_vae_run.sh "${file}" "${output_directory}"
done
