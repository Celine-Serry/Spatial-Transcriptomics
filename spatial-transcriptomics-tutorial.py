# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:08:38 2024

@author: Celin
"""

#Spatial transcriptomics 

#Tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
#Tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/integration-scanorama.html

#%%import packages
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%Read in data
#The function datasets.visium_sge() downloads the dataset from 10x Genomics and returns an AnnData object 
#that contains counts, images and spatial coordinates. We will calculate standards QC metrics with 
#pp.calculate_qc_metrics and percentage of mitochondrial read counts per sample.

#When using your own Visium data, use sc.read_visium() function to import it.

adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

#How data looks
adata

#%%QC and preprocessing 
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
sns.histplot(
    adata.obs["total_counts"][adata.obs["total_counts"] < 10000],
    kde=False,
    bins=40,
    ax=axs[1],
)
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
sns.histplot(
    adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000],
    kde=False,
    bins=60,
    ax=axs[3],
)

#Filter 
sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=10)

#Normalize data
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

#%% Manifold embedding and clustering based on transcriptional similarity

#To embed and cluster the manifold encoded by transcriptional similarity, we proceed as in the standard clustering tutorial.

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(
    adata, key_added="clusters",  directed=False, n_iterations=2
)

#We plot some covariates to check if there is any particular structure in the UMAP associated with total counts and detected genes.
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)

#%% Visualization in spatial coordinates

#Let us now take a look at how total_counts and n_genes_by_counts behave in spatial coordinates. 
#We will overlay the circular spots on top of the Hematoxylin and eosin stain (H&E) image provided, 
#using the function sc.pl.spatial.

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"])

#The function sc.pl.spatial accepts 4 additional parameters:
#img_key: key where the img is stored in the adata.uns element
#crop_coord: coordinates to use for cropping (left, right, top, bottom)
#alpha_img: alpha value for the transcparency of the image
#bw: flag to convert the image into gray scale
#Furthermore, in sc.pl.spatial, the size parameter changes its behaviour: it becomes a scaling factor for the spot sizes.

#Before, we performed clustering in gene expression space, and visualized the results with UMAP. 
#By visualizing clustered samples in spatial dimensions, we can gain insights into tissue organization and, 
#potentially, into inter-cellular communication.

sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5)

#Spots belonging to the same cluster in gene expression space often co-occur in spatial dimensions. 
#For instance, spots belonging to cluster 5 are often surrounded by spots belonging to cluster 0.

#We can zoom in specific regions of interests to gain qualitative insights. 
#Furthermore, by changing the alpha values of the spots, we can visualize better the underlying tissue 
#morphology from the H&E image.

sc.pl.spatial(
    adata,
    img_key="hires",
    color="clusters",
    groups=["5", "9"],
    crop_coord=[7000, 10000, 0, 6000],
    alpha=0.5,
    size=1.3,
)

#%% Cluster marker genes

#Let us further inspect cluster 5, which occurs in small groups of spots across the image.

#Compute marker genes and plot a heatmap with expression levels of its top 10 marker genes across clusters.
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")
sc.pl.rank_genes_groups_heatmap(adata, groups="9", n_genes=10, groupby="clusters")

#We see that CR2 recapitulates the spatial structure.

sc.pl.spatial(adata, img_key="hires", color=["clusters", "CR2"])
sc.pl.spatial(adata, img_key="hires", color=["COL1A2", "SYPL1"], alpha=0.7)

#%% Integration with single cell RNA seq

#%% Import packages
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
from pathlib import Path

#%% Read in data

#We will use two Visium spatial transcriptomics dataset of the mouse brain (Sagittal), 
#which are publicly available from the 10x genomics website.

#The function datasets.visium_sge() downloads the dataset from 10x genomics and returns an AnnData object 
#that contains counts, images and spatial coordinates. We will calculate standards QC metrics with pp.calculate_qc_metrics and visualize them.

#When using your own Visium data, use Scanpy’s read_visium() function to import it.
adata_spatial_anterior = sc.datasets.visium_sge(
    sample_id="V1_Mouse_Brain_Sagittal_Anterior"
)
adata_spatial_posterior = sc.datasets.visium_sge(
    sample_id="V1_Mouse_Brain_Sagittal_Posterior"
)

adata_spatial_anterior.var_names_make_unique()
adata_spatial_posterior.var_names_make_unique()
sc.pp.calculate_qc_metrics(adata_spatial_anterior, inplace=True)
sc.pp.calculate_qc_metrics(adata_spatial_posterior, inplace=True)

#%% Quality control 

for name, adata in [
    ("anterior", adata_spatial_anterior),
    ("posterior", adata_spatial_posterior),
]:
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle(f"Covariates for filtering: {name}")

    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(
        adata.obs["total_counts"][adata.obs["total_counts"] < 20000],
        kde=False,
        bins=40,
        ax=axs[1],
    )
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
    sns.histplot(
        adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000],
        kde=False,
        bins=60,
        ax=axs[3],
    )
    

#sc.datasets.visium_sge downloads the filtered visium dataset, the output of spaceranger that contains only 
#spots within the tissue slice. Indeed, looking at standard QC metrics we can observe that the samples 
#do not contain empty spots.

#We proceed to normalize Visium counts data with the built-in normalize_total method from Scanpy, 
#and detect highly-variable genes (for later). As discussed previously, 
#note that there are more sensible alternatives for normalization (see discussion in sc-tutorial paper and 
#more recent alternatives such as SCTransform or GLM-PCA).

for adata in [
    adata_spatial_anterior,
    adata_spatial_posterior,
]:
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, inplace=True)
    
#%% Data Integration

adatas = [adata_spatial_anterior, adata_spatial_posterior]
adatas_cor = scanorama.correct_scanpy(adatas, return_dimred=True)

#We will concatenate the two datasets and save the integrated embeddings in adata_spatial.obsm['scanorama_embedding']. 
#Furthermore we will compute UMAP to visualize the results and qualitatively assess the data integration task.

#Notice that we are concatenating the two dataset with uns_merge="unique" strategy, 
#in order to keep both images from the visium datasets in the concatenated anndata object.

adata_spatial = sc.concat(
    adatas_cor,
    label="library_id",
    uns_merge="unique",
    keys=[
        k
        for d in [
            adatas_cor[0].uns["spatial"],
            adatas_cor[1].uns["spatial"],
        ]
        for k, v in d.items()
    ],
    index_unique="-",
)

sc.pp.neighbors(adata_spatial, use_rep="X_scanorama")
sc.tl.umap(adata_spatial)
sc.tl.leiden(
    adata_spatial, key_added="clusters", n_iterations=2, directed=False
)

sc.pl.umap(
    adata_spatial, color=["clusters", "library_id"], palette=sc.pl.palettes.default_20
)

#We can also visualize the clustering result in spatial coordinates. 
#For that, we first need to save the cluster colors in a dictionary. 
#We can then plot the Visium tissue fo the Anterior and Posterior Sagittal view, alongside each other.
clusters_colors = dict(
    zip([str(i) for i in range(18)], adata_spatial.uns["clusters_colors"])
)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))

for i, library in enumerate(
    ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]
):
    ad = adata_spatial[adata_spatial.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad,
        img_key="hires",
        library_id=library,
        color="clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist()
        ],
        legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()

#From the clusters, we can clearly see the stratification of the cortical layer in both of the tissues 
#(see the Allen brain atlas for reference). Furthermore, it seems that the dataset integration worked well, 
#since there is a clear continuity between clusters in the two tissues.

#%% Data integration and label transfer from scRNA-seq dataset

#We can also perform data integration between one scRNA-seq dataset and one spatial transcriptomics dataset. 
#Such task is particularly useful because it allows us to transfer cell type labels to the Visium dataset, 
#which were dentified from the scRNA-seq dataset.

#For this task, we will be using a dataset from Tasic et al., where the mouse cortex was profiled with 
#smart-seq technology.

#The dataset can be downloaded from GEO count - metadata. 
#Conveniently, you can also download the pre-processed dataset in h5ad format from here.

#Since the dataset was generated from the mouse cortex, 
#we will subset the visium dataset in order to select only the spots part of the cortex. 
#Note that the integration can also be performed on the whole brain slice, 
#but it would give rise to false positive cell type assignments and and therefore it should be interpreted 
#with more care.

#The integration task will be performed with Scanorama: each Visium dataset will be integrated with the 
#smart-seq cortex dataset.

#The following cell should be uncommented out and run if this is the first time running this notebook.
if not Path("./data/adata_processed.h5ad").exists():
    !wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746_cells_exon_counts.csv.gz -O data/GSE115746_cells_exon_counts.csv.gz
    !gunzip data/GSE115746_cells_exon_counts.csv.gz
    !wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746_complete_metadata_28706-cells.csv.gz -O data/GSE115746_complete_metadata_28706-cells.csv.gz
    !gunzip data/GSE115746_complete_metadata_28706-cells.csv.gz
    %pip install pybiomart
    counts = pd.read_csv("data/GSE115746_cells_exon_counts.csv", index_col=0).T
    meta = pd.read_csv(
        "data/GSE115746_complete_metadata_28706-cells.csv", index_col="sample_name"
    )
    meta = meta.loc[counts.index]
    annot = sc.queries.biomart_annotations(
        "mmusculus",
        ["mgi_symbol", "ensembl_gene_id"],
    ).set_index("mgi_symbol")
    annot = annot[annot.index.isin(counts.columns)]
    counts = counts.rename(columns=dict(zip(annot.index, annot["ensembl_gene_id"])))
    adata_cortex = an.AnnData(counts, obs=meta)
    sc.pp.normalize_total(adata_cortex, inplace=True)
    sc.pp.log1p(adata_cortex)
    adata_cortex.write_h5ad("data/adata_processed.h5ad")
    
    
#if you cannot run wget, use this:
import os
import pandas as pd
import requests
import scanpy as sc
import anndata as an
from pathlib import Path

# Define function to download and decompress files
def download_and_decompress(url, filename):
    if not Path(filename).exists():
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

# Define URLs
url_counts = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746_cells_exon_counts.csv.gz"
url_metadata = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746_complete_metadata_28706-cells.csv.gz"

# Define filenames
counts_filename = "data/GSE115746_cells_exon_counts.csv.gz"
metadata_filename = "data/GSE115746_complete_metadata_28706-cells.csv.gz"

# Download and decompress files
download_and_decompress(url_counts, counts_filename)
download_and_decompress(url_metadata, metadata_filename)

# Read data into pandas DataFrame
counts = pd.read_csv(counts_filename, index_col=0).T
meta = pd.read_csv(metadata_filename, index_col="sample_name")

# Filter metadata based on counts index
meta = meta.loc[counts.index]

# Additional steps for annotation
annot = sc.queries.biomart_annotations(
       "mmusculus",
       ["mgi_symbol", "ensembl_gene_id"],
   ).set_index("mgi_symbol")
annot = annot[annot.index.isin(counts.columns)]
counts = counts.rename(columns=dict(zip(annot.index, annot["ensembl_gene_id"])))
adata_cortex = an.AnnData(counts, obs=meta)
sc.pp.normalize_total(adata_cortex, inplace=True)
sc.pp.log1p(adata_cortex)
adata_cortex.write_h5ad("data/adata_processed.h5ad")

# Further processing and analysis
# ...

# Save processed data
if not Path("./data/adata_processed.h5ad").exists():
    # Preprocessing steps...
    # ...
    adata_cortex.write_h5ad("data/adata_processed.h5ad")


########REMOVED CODE ################
#%% [6] Spatially variable genes
#identify patterns of gene expression #pip install spatialde 

#explanation here: https://github.com/Teichlab/SpatialDE

#check spatial statistics
counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names,index=adata.obs_names)
coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
results = SpatialDE.run(coord, counts)

# Convert Pandas DataFrame to NumPy array
coord_array = coord.to_numpy()
counts_array = counts.to_numpy()

# Now run the function with the NumPy arrays
results = SpatialDE.run(coord_array, counts_array)

# Convert NumPy arrays back to Pandas DataFrames
coord_df = pd.DataFrame(coord_array, index=coord.index, columns=coord.columns)
counts_df = pd.DataFrame(counts_array, index=counts.index, columns=counts.columns)

# Now run the function with the Pandas DataFrames
results = SpatialDE.run(coord_df, counts_df)


#%% [5] Decide on cell types 

#score in column gives how much it looks like a certain annotated cell type

#take the highest score of those columns 

#make list of all cell types
cell_types = ['astrocyte', 'caudal ganglionic eminence derived interneuron', 'endothelial cell', 'glutamatergic neuron', 
              'inhibitory interneuron', 'medial ganglionic eminence derived interneuron', 'microglial cell', 
              'neural progenitor cell', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'pericyte', 'radial glial cell', 
              'vascular associated smooth muscle cell']

max_cell_type1 = adata_1_transfer.obs[cell_types].idxmax(axis=1)
adata_1_transfer.obs['cell_type'] = max_cell_type1

max_cell_type2 = adata_1_transfer.obs[cell_types].drop(columns=['radial glial cell']).idxmax(axis=1)
adata_1_transfer.obs['cell_type2'] = max_cell_type2 


df_1 = adata_1_transfer.obs

#or maybe only take the main cell types
cell_types = ['astrocyte', 'caudal ganglionic eminence derived interneuron', 'endothelial cell', 'glutamatergic neuron', 
              'inhibitory interneuron', 'medial ganglionic eminence derived interneuron', 'microglial cell', 
              'oligodendrocyte']

max_cell_type3 = adata_1_transfer.obs[cell_types].idxmax(axis=1)
adata_1_transfer.obs['cell_type3'] = max_cell_type3 

df_1 = adata_1_transfer.obs


for i in cell_types:
    sc.pl.spatial(adata_1_transfer, img_key="hires", color=["cell_type", i], size=1.5)
    sc.pl.spatial(adata_2_transfer, img_key="hires", color=["cell_type", i], size=1.5)

adata_1_transfer.var_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in adata_1_transfer.var_names]
adata_2_transfer.var_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in adata_2_transfer.var_names]

sc.pl.spatial(adata_1_transfer, img_key="hires", color=["cell_type"], size=1.5)
sc.pl.spatial(adata_2_transfer, img_key="hires", color=["cell_type"], size=1.5)

#To zoom 
sc.pl.spatial(
    adata_2_transfer,
    img_key="hires",
    color="radial glial cell",
    crop_coord=[5000, 11000, 5000, 11000],
    alpha=0.5,
    size=1.3,
)



#%% Find out the major composition of the cell types per cluster - not right 

# Group by Leiden clusters and calculate the sum of each cell type in each cluster
cluster_celltype_counts1 = adata_1_transfer.obs.groupby('leiden')[cell_types].sum()
cluster_celltype_counts2 = adata_2_transfer.obs.groupby('leiden')[cell_types].sum()

# Calculate the total number of cells in each cluster
cluster_total_cells1 = cluster_celltype_counts1.sum(axis=1)
cluster_total_cells2 = cluster_celltype_counts2.sum(axis=1)

# Calculate the proportion of each cell type in each cluster
cluster_celltype_percentage1 = (cluster_celltype_counts1.div(cluster_total_cells1, axis=0)) * 100
cluster_celltype_percentage2 = (cluster_celltype_counts2.div(cluster_total_cells2, axis=0)) * 100 

# Print the proportions of each cell type in each cluster
print(cluster_celltype_percentage1)
print(cluster_celltype_percentage2)

#stacked bar plot
import matplotlib.pyplot as plt

cluster_celltype_percentage1.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Cell Types in Each Leiden Cluster')
plt.xlabel('Leiden Cluster')
plt.ylabel('%')
plt.xticks(rotation=0)
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

cluster_celltype_percentage2.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Cell Types in Each Leiden Cluster')
plt.xlabel('Leiden Cluster')
plt.ylabel('%')
plt.xticks(rotation=0)
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#Check whether the cell types in the different clusters differ from each other 
#i.e., compare astrocytes of leiden cluster 1 to astrocytes of leiden cluster 2 

#first make column with a combi of cell type and cluster (i.e., astro1, astro2 etc.)
adata_1_transfer
adata_2_transfer

cell_types = ['astrocyte', 'caudal ganglionic eminence derived interneuron', 'endothelial cell', 
              'glutamatergic neuron', 'inhibitory interneuron', 
              'medial ganglionic eminence derived interneuron', 'microglial cell', 
              'neural progenitor cell', 'oligodendrocyte', 'oligodendrocyte precursor cell',
              'pericyte', 'radial glial cell', 'vascular associated smooth muscle cell']

leiden_clusters_1 = adata_1_transfer.obs['leiden'].unique().tolist()
leiden_clusters_2 = adata_2_transfer.obs['leiden'].unique().tolist()

#empty list to store labels
combined_labels = []

#go over every row 
for index, row in adata_2_transfer.obs.iterrows():
    #get the cell type from the column name 
    cell_type = row.index[row.index.str.startswith('astrocyte')].values[0]
    #get leiden cluster of current row
    leiden_cluster = str(row['leiden'])
    #combine the two values
    combined_label = f"{cell_type}_cluster_{leiden_cluster}"
    #append the combined label to the list
    combined_labels.append(combined_label)
#add the combined label as new column in the df
adata_2_transfer.obs['cell_type_cluster'] = combined_labels

#empty list to store labels
combined_labels = []

#go over every row 
for index, row in adata_1_transfer.obs.iterrows():
    #get the cell type from the column name 
    cell_type = row.index[row.index.str.startswith('astrocyte')].values[0]
    #get leiden cluster of current row
    leiden_cluster = str(row['leiden'])
    #combine the two values
    combined_label = f"{cell_type}_cluster_{leiden_cluster}"
    #append the combined label to the list
    combined_labels.append(combined_label)
#add the combined label as new column in the df
adata_1_transfer.obs['cell_type_cluster'] = combined_labels