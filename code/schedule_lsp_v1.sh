

# create a list of files
files=(pancreatic-adeno+lung-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+colon-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+glioblastoma_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+lung-adeno_transcriptomics_cell-line+CPTAC.tsv renal-clear-cell+pancreatic-adeno_transcriptomics_cell-line+CPTAC.tsv)


# iterate thorugh files
for file in "${files[@]}"; do
  # iterate thorugh the pair wise combination of cancer and system weights
    echo file="${file}"
    sbatch ./create_lsp_v1.sh 50 64 1500 ../output/cancer_pairs/${file}
done
