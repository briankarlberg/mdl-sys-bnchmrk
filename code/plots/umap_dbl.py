import umap
import numpy as np
import pandas as pd

systems_column = "System"
cancer_column = "Cancer_type"

data = pd.read_csv(
    '../results/dbl/colon-nos-adeno_transcriptomics_pancreatic-nos-ductal-ad_transcriptomics_HCMI+CPTAC.latent_space.tsv',
    sep="\t", index_col=0)

# remove the systems column and cancer column
cleaned_df = data.drop(columns=[systems_column, cancer_column])
cancers = data[cancer_column]
systems = data[systems_column]
print(cleaned_df)

umap = umap.UMAP()

umap_df = umap.fit_transform(cleaned_df)

# plot the umap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.scatterplot(x=umap_df[:, 0], y=umap_df[:, 1], hue=cancers, style=systems,
                palette=sns.color_palette("hsv", len(cancers.unique())), legend='full')
plt.title('UMAP projection colored by {}'.format(cancer_column))
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
# push the legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.show()
