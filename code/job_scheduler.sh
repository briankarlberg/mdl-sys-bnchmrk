#pancreatic-adeno+lung-adeno_transcriptomics_cell-line+CPTAC.tsv
#renal-clear-cell+colon-adeno_transcriptomics_cell-line+CPTAC.tsv
#renal-clear-cell+glioblastoma_transcriptomics_cell-line+CPTAC.tsv
#renal-clear-cell+lung-adeno_transcriptomics_cell-line+CPTAC.tsv
#renal-clear-cell+pancreatic-adeno_transcriptomics_cell-line+CPTAC.tsv

# create a list of files
files=(pancreatic-adeno+lung-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+colon-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+glioblastoma_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+lung-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+pancreatic-adeno_transcriptomics_cell-line+CPTAC.tsv)
cancer_weights=(2 2.5 3 3.5 4)
system_weights=(2 2.5 3 3.5 4)


# iterate thorugh files
for file in "${files[@]}"; do
  # iterate thorugh the pair wise combination of cancer and system weights
  for i in $(seq 0 4); do
    for j in $(seq 0 4); do
      echo file="${file}" cancer_weight="${cancer_weights[i]}" system_weight="${system_weights[j]}"
      sbatch ./schedule_vae_cpu.sh 50 64 1500 ../output/cancer_pairs/${file}  ../results/ --cancer_weight "${cancer_weights[i]}" --system_weight "${system_weights[j]}"
    done
  done
done
