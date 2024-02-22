# strctr_one.py
# Script to build one_cncr files
    # Sript for building strctrd/two_cncr files
    # will read from strctrd/one_cncr files
"""
# Download data

# pwd: data/beataml
# cd.download_data_by_prefix('beataml')

# pwd: data/cell_line
# cd.download_data_by_prefix('cell_line')

# pwd: data/cptac
# cd.download_data_by_prefix('cptac')

# pwd: data/hcmi
# cd.download_data_by_prefix('hcmi')
"""
# pwd: /mdl-sys-bnchmrk/code

Imports
import coderdata as cd
import glob
import pandas as pd
import umap
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# pwd
# mdl-sys-bnchmrk/code

# ls
# strctr.py

# Import functions
import strctr

systems = 'cell-line+CPTAC'

# Read data
cell_line = cd.DatasetLoader('cell_line', data_directory = '../data/cell_line/') # a
cptac = cd.DatasetLoader('cptac', data_directory = '../data/cptac/') # b

# Cell lines are system A
sys_a_samp = cell_line.samples
sys_a = 'cell-line'
sys_a_lbl = 'cell_line'

# CPTAC is system B
sys_b_samp = cptac.samples
sys_b = 'cptac'
sys_b_lbl = 'CPTAC'

# BeatAML is system B
#

# Transcriptomics data modality extraction
modality = 'transcriptomics' # to file name
moda = 'tran_' # to columns and index
mda_n_sys_a = cell_line.transcriptomics[cell_line.transcriptomics.improve_sample_id.isin(ids_sys_a)] # cl
mda_n_sys_b = cptac.transcriptomics[cptac.transcriptomics.improve_sample_id.isin(ids_sys_b)]

# Proteomics data modality extraction
# modality = 'proteomics' # to file name
# moda = 'prot_' # to columns and index
# mda_n_sys_a = cell_line.proteomics[cell_line.proteomics.improve_sample_id.isin(ids_proj_a)] # cl
# mda_n_sys_b = cptac.proteomics[cptac.proteomics.improve_sample_id.isin(ids_proj_b)]

# Copy number and Mutations data modality extractions insertion point

names = ['breast-ductal', 'breast-lobular', 'breast-nos']
labels = ['breast_ductal', 'breast_lobular', 'breast_nos']
a_list = ['Breast Invasive Ductal Carcinoma',
          'Breast Invasive Lobular Carcinoma',
          'Breast Invasive Carcinoma, NOS']
b_list = ['Breast carcinoma',
          'Breast carcinoma',
          'Breast carcinoma']

print(cncr)
ids_sys_a = sys_a_samp_canc_n.improve_sample_id # cl
ids_sys_b = sys_b_samp_canc_n.improve_sample_id # cp

for i, cncr in enumerate(names):
    # cncr = a_list[i]
    cncr_lbl = labels[i]
    sys_a_samp_canc_n = sys_a_samp[sys_a_samp.cancer_type == a_list[i]]
    sys_b_samp_canc_n = sys_b_samp[sys_b_samp.cancer_type == b_list[i]]

    df_lite, size, na_count, inf_count = df_check(mda_n_sys_a)
    print(sys_a, '| sys a')
    print(cncr, modality)
    print('len: ', size)
    print('NaNs: ', na_count)
    print('Infs: ', inf_count)

    wall_clock, dot_T = extract(df_lite)
    # print(wall_clock)
    # dot_T.iloc[:3, :3]
    #
    # dot_T = g(moda, dot_T)
    dot_T = g(moda, dot_T.copy())
    dot_T.dropna(axis = 1, inplace = True)
    a = dot_T # cell line

    df_lite, size, na_count, inf_count = df_check(mda_n_sys_b)
    print(sys_b, '| sys b')
    print(cncr, modality)
    print('len: ', size)
    print('NaNs: ', na_count)
    print('Infs: ', inf_count)
    
    wall_clock, dot_T = extract(df_lite)
    dot_T = g(moda, dot_T.copy())
    dot_T.dropna(axis = 1, inplace = True)
    b = dot_T # cptac
    
    a.insert(0, 'Cancer_type', cncr_lbl)
    b.insert(0, 'Cancer_type', cncr_lbl)
    a.insert(0, 'System', sys_a_lbl)
    b.insert(0, 'System', sys_b_lbl)

    ab = pd.concat([a, b], axis=0, join='inner')
    print(ab.System.value_counts())
    print(ab.Cancer_type.value_counts())

    out_one = '../strctrd/one_cncr/'

    # Write two-system, single cancer type to disk
    ab.to_csv(
        '../strctrd/'+out_one+'/'+cncr+'_'+modality+'_'+systems+'.tsv',
        sep = '\t')
    # break