# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:21:03 2024

@author: Celin
"""

#Spatial transcriptomics 

#%%import packages
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import SpatialDE
import NaiveDE
import scipy
import scipy.sparse


#%% Settings

sc.set_figure_params(facecolor='white', figsize=(8,8))
sc.settings.verbosity=3

#%% Tutorial 1: One Spatial Transcriptomics dataset 

#%% [1] Read in data
#The function datasets.visium_sge() downloads the dataset from 10x Genomics and returns an AnnData object 
#that contains counts, images and spatial coordinates. We will calculate standards QC metrics with 
#pp.calculate_qc_metrics and percentage of mitochondrial read counts per sample.

#When using your own Visium data, use sc.read_visium() function to import it.

adata = sc.datasets.visium_sge(sample_id="Targeted_Visium_Human_Cerebellum_Neuroscience")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

#How data looks
adata

#%% [2] QC and preprocessing 

fig, axs = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[1])

#Filter 
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=10)

#%% [3] Normalize data
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
#sc.pp.scale(adata) #can choose to scale the data. 

#%% [4] PCA, UMAP, clustering 

#To embed and cluster the manifold encoded by transcriptional similarity, we proceed as in the standard clustering tutorial.
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters",  directed=False, n_iterations=2)

#We plot some covariates to check if there is any particular structure in the UMAP associated with total counts and detected genes.
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)

#can try different resolutions: 
sc.tl.leiden(adata, resolution=0.8, key_added="clusters_0.8",  directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=0.6, key_added="clusters_0.6",  directed=False, n_iterations=2)
sc.tl.leiden(adata, resolution=0.4, key_added="clusters_0.4",  directed=False, n_iterations=2)

sc.pl.umap(adata, color=["clusters_0.4", "clusters_0.6", "clusters_0.8"], wspace=0.4)

#%% [5] Visualization in spatial coordinates

#Let us now take a look at how total_counts and n_genes_by_counts behave in spatial coordinates. 
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
sc.pl.spatial(adata, img_key=None, color="clusters_0.8", size=1.5)
sc.pl.spatial(adata, img_key=None, color="clusters_0.4", size=1.5)

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

#%% Plot marker genes in UMAP

#make dictionary of marker genes:
marker_genes = {
    "Neuron": ["clusters_0.4", "VIP", "GAD1", "GAD2", "RELN", "CNR1"],
    "Astrocyte": ["clusters_0.4", "GFAP", "AQP4", "SLC1A2", "GJB6", "GJA1"],
    "Oligodendrocyte": ["clusters_0.4", "MOG", "MBP", "MAG", "CNP", "PLP1", "UGT8"],
    "Microglia": ["clusters_0.4",  "CX3CR1", "CSF1R"],
    "OPC": ["clusters_0.4", "OLIG2", "SOX10", "PDGFRA"],
    "Endothelial Cell":["clusters_0.4", "ABCB1"]
}


#plot UMAP for each cell type's marker genes
for cell_type, genes in marker_genes.items():
    plt.figure(figsize=(8, 6))
    sc.pl.spatial(adata, img_key=None, color=genes, show=False, use_raw=False, size=1.5)
    plt.suptitle(cell_type, fontsize=16)
    plt.show()
    
#%% [5] DGE in clusters

#Compute marker genes and plot a heatmap with expression levels of its top 10 marker genes across clusters.
sc.tl.rank_genes_groups(adata, "clusters_0.4", method="t-test")
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby="clusters_0.4")

def extract_genes(adata, number_groups):
    num_groups = number_groups
    genes_groups = {}

    genes_df = adata.uns['rank_genes_groups']['names']
    for i in range(num_groups):
        # Extract the genes associated with the current group
        group_genes = genes_df[str(i)]
        
        # Assign the genes to a global variable
        globals()["genes_group_" + str(i)] = group_genes
        
        # Store the genes in the dictionary
        genes_groups["genes_group_" + str(i)] = group_genes
    
    return genes_group_0

#example use 
extract_genes(adata=adata, number_groups=3)

#now you can access them like:
genes_group_0
genes_group_1
genes_group_2
    

#%% [6] Spatially variable genes
#identify patterns of gene expression #pip install spatialde 
#explanation here: https://github.com/Teichlab/SpatialDE

#transform data into correct format
counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names, index=adata.obs_names)
    
#get sample information from adata
sample_info = adata.obs[['array_row', 'array_col', 'total_counts']].copy()
sample_info.columns = ['x', 'y', 'total_counts']

#plot the x and y coordinates to see which locations of the tissue slice have been sampled
plt.figure(figsize=(6, 4))
plt.scatter(sample_info['x'], sample_info['y'], c='k', s=5)
plt.axis('equal')
plt.show()

#normalize data 
norm_expr = NaiveDE.stabilize(counts.T).T

#regress out sequencing depth effect
resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T
sample_resid_expr = resid_expr.sample(n=1000, axis=1, random_state=1)

#get spatial coordinates
X = sample_info[['x', 'y']].values  
#run SpatialDE
results = SpatialDE.run(X, resid_expr)

#NOTE: in util.py from spatialDE, add import numpy as np and change all sp. to np.
#else get error on cannot import arrange from scipy (which is a numpy function).

#The most important columns are
#g - The name of the gene
#pval - The P-value for spatial differential expression
#qval - Significance after correcting for multiple testing
#l - A parameter indicating the distance scale a gene changes expression over

# Display results
print(results.head())

results.sort_values('qval').head(10)[['g', 'l', 'qval']]

#We detected a few spatially differentially expressed genes, GFAP, ATP1A3 and MBP for example.

#A simple way to visualize these genes is by plotting the x and y coordinates as above, 
#but letting the color correspond to expression level.

plt.figure(figsize=(7, 4))
for i, g in enumerate(['GFAP', 'ATP1A3', 'MBP']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=norm_expr[g], s=5);
    plt.title(g)
    plt.axis('equal')


    plt.colorbar(ticks=[]);

#For reference, we can compare these to genes which are not spatially DE
results.sort_values('qval').tail(10)[['g', 'l', 'qval']]

#PRKN, ACTN1, ATP6V1D
plt.figure(figsize=(7, 4))
for i, g in enumerate(['PRKN', 'ACTN1', 'ATP6V1D']):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=norm_expr[g], s=5);
    plt.title(g)
    plt.axis('equal')


    plt.colorbar(ticks=[]);

#In regular differential expression analysis, we usually investigate the relation between 
#significance and effect size by so called volcano plots. We don't have the concept of 
#fold change in our case, but we can investigate the fraction of variance explained by spatial variation.
plt.figure(figsize=(5, 4))
plt.yscale('log')
plt.scatter(results['FSV'], results['qval'], c='black')
plt.axhline(0.05, c='black', lw=1, ls='--');
plt.gca().invert_yaxis();
plt.xlabel('Fraction spatial variance')
plt.ylabel('Adj. P-value');

#Automatic expression histology 
#To perform automatic expression histology (AEH), the genes should be filtered by SpatialDE significance. 
#For this example, let us use a very weak threshold. But in typical use, filter by qval < 0.05
sign_results = results.query('qval < 0.05')

#AEH requires two parameters: the number of patterns, and the characteristic lengthscale for histological patterns.
#For some guidance in picking the lengthscale l we can look at the optimal lengthscale 
#for the signficant genes.
lengthscale = sign_results['l'].value_counts()
lengthscale= pd.DataFrame(lengthscale)
lengthscale['total'] = lengthscale.index * lengthscale['count']
average = sum(lengthscale['total'] / sum(lengthscale['count'])) #10.43 
#to use some extra spatial covariance, we put this paramater to l = 10.8.
#For the number of patterns, we try C = 3.
histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr, sign_results, C=3, l=10.8, verbosity=1)

#After fitting the AEH model, the function returns two DataFrames, 
#one with pattern membership information for each gene:
histology_results.head()

#And one with realizations for the underlying expression for each histological pattern.

#We can visualize this underlying expression in the tissue context as we would for any individual gene.
plt.figure(figsize=(10, 3))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(sample_info['x'], sample_info['y'], c=patterns[i], s=5);
    plt.axis('equal')
    plt.title('Pattern {} - {} genes'.format(i, histology_results.query('pattern == @i').shape[0] ))
    plt.colorbar(ticks=[]);
    
#It is usually interesting to see what the coexpressed genes determining a histological pattern are:
for i in histology_results.sort_values('pattern').pattern.unique():
    print('Pattern {}'.format(i))
    print(', '.join(histology_results.query('pattern == @i').sort_values('membership')['g'].tolist()))
    print()

#lets save all results 
adata.write_h5ad("E:/Major project/adata_spatial_5-20-2024.h5ad")
results2.to_csv("E:/Major project/spatialDE_results.csv")
histology_results.to_csv("E:/Major project/spatialDE_histology_results.csv")

#read in old results
results_old = pd.read_csv("D:/Major project/spatialDE_results.csv")
histology_old = pd.read_csv("D:/Major project/spatialDE_histology_results.csv")

#%% Tutorial 2: Integration of two spatial transcriptomics datasets

#%% [1] Import packages
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
from pathlib import Path

#%% [2] Read in data

#We will use two Visium spatial transcriptomics dataset of the mouse brain (Sagittal), 
#which are publicly available from the 10x genomics website.

#The function datasets.visium_sge() downloads the dataset from 10x genomics and returns an AnnData object 
#that contains counts, images and spatial coordinates. We will calculate standards QC metrics with pp.calculate_qc_metrics and visualize them.

#When using your own Visium data, use Scanpy’s read_visium() function to import it.

#https://www.10xgenomics.com/datasets/adult-human-brain-1-cerebral-cortex-unknown-orientation-stains-anti-gfap-anti-nfh-1-standard-1-1-0
#https://www.10xgenomics.com/datasets/adult-human-brain-2-cerebral-cortex-unknown-orientation-stains-anti-snap-25-anti-nfh-1-standard-1-1-0

adata_spatial_1 = sc.datasets.visium_sge(
    sample_id="V1_Human_Brain_Section_1"
)
adata_spatial_2 = sc.datasets.visium_sge(
    sample_id="V1_Human_Brain_Section_2"
)

#sc.datasets.visium_sge downloads the filtered visium dataset, the output of spaceranger that contains only 
#spots within the tissue slice. Indeed, looking at standard QC metrics we can observe that the samples 
#do not contain empty spots.

#need to run var names make unique 
for adata in [
    adata_spatial_1,
    adata_spatial_2,
]:
    adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)

#%% [3] Investigate our data

#we start of with 
adata_spatial_1 #n_obs × n_vars = 4910 × 36601
adata_spatial_2 #n_obs × n_vars = 4972 × 36601

#shape should correspond to the number of positions (n_obs)
adata_spatial_1.obsm['spatial'].shape #(4910, 2)
adata_spatial_2.obsm['spatial'].shape #(4972, 2)

#current columns in adata.obs
adata_spatial_1.obs
adata_spatial_2.obs

#add a dummy variable to adata.obs: add column of just a's so we can look at where positions were mapped onto the tissue 
#allows you to see which variables are removed after the filtering 
adata_spatial_1.obs['thing'] = 'a'
adata_spatial_2.obs['thing'] = 'a'

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_1, color = 'thing')

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_2, color = 'thing')


#%% [4] Quality control 

for name, adata in [
    ("1", adata_spatial_1),
    ("2", adata_spatial_2),
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
    
#to zoom in more closely to the plots:
sns.distplot(adata_spatial_1.obs["total_counts"][adata_spatial_1.obs["total_counts"] < 2000], kde=False, bins=40)
sns.distplot(adata_spatial_2.obs["total_counts"][adata_spatial_2.obs["total_counts"] < 2000], kde=False, bins=40)

sns.distplot(adata_spatial_1.obs["total_counts"][adata_spatial_1.obs["total_counts"] > 15000], kde=False, bins=40)
sns.distplot(adata_spatial_2.obs["total_counts"][adata_spatial_2.obs["total_counts"] > 15000], kde=False, bins=40)

sns.distplot(adata_spatial_1.obs["n_genes_by_counts"][adata_spatial_1.obs["n_genes_by_counts"] < 2000], kde=False, bins=40)
sns.distplot(adata_spatial_2.obs["n_genes_by_counts"][adata_spatial_2.obs["n_genes_by_counts"] < 2000], kde=False, bins=40)

sns.distplot(adata_spatial_1.obs["n_genes_by_counts"][adata_spatial_1.obs["n_genes_by_counts"] > 4000], kde=False, bins=40)
sns.distplot(adata_spatial_2.obs["n_genes_by_counts"][adata_spatial_2.obs["n_genes_by_counts"] > 4000], kde=False, bins=40)

for adata in [
    adata_spatial_1,
    adata_spatial_2,
]:
    #calculate % of mitochondrial reads
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    #check % of mitochondrial reads
    sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4)


#%% [5] Filtering 

#Perform actual filtering (already done for the sc.datasets.visium.sge(), but if you have your own dataset, then you want
#to run this chunk of code too)

for adata in [
    adata_spatial_1,
    adata_spatial_2,
]:
    #Filter cell outliers based on counts and numbers of genes expressed: filter cells if number of genes is lower than 1000 or higher than 35000
    sc.pp.filter_cells(adata, min_counts = 1000) 
    sc.pp.filter_cells(adata, max_counts=35000) 
    #Filter genes based on number of cells or counts: filter genes that arent in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3) #n_vars from 36601 to 22007

#Filter cells based on amount of mitochondrial reads
#adata_spatial_1 = adata_spatial_1[adata_spatial_1.obs["pct_counts_mt"] < 20]
#adata_spatial_2 = adata_spatial_2[adata_spatial_2.obs["pct_counts_mt"] < 20]

#check which cells you filtered  (could be dead cells or sequencing artifacts)
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_1, color = 'thing')

adata_spatial_2.obs['thing'] = 'a'
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_2, color = 'thing')

#%% [6] Log-normalize and find highly variable genes 
#We proceed to normalize Visium counts data with the built-in normalize_total method from Scanpy, 
#and detect highly-variable genes (for later). As discussed previously, 
#note that there are more sensible alternatives for normalization (see discussion in sc-tutorial paper and 
#more recent alternatives such as SCTransform or GLM-PCA).

for adata in [
    adata_spatial_1,
    adata_spatial_2,
]:
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, inplace=True)

#lets save the data now for later usage
adata_spatial_1.write_h5ad("D:/Major project/data/adata_spatial_1_lognorm.h5ad")
adata_spatial_2.write_h5ad("D:/Major project/data/adata_spatial_2_lognorm.h5ad")

#%% [7] UMAP 

#read in data
adata_spatial_1 = sc.read_h5ad("D:/Major project/data/adata_spatial_1_lognorm.h5ad")
adata_spatial_2 = sc.read_h5ad("D:/Major project/data/adata_spatial_2_lognorm.h5ad")

for adata in [
    adata_spatial_1,
    adata_spatial_2,
]:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

#you can tell something is wrong in this QC becuase counts are biased by location/cluster
#may want to reduce threshold to remove
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata_spatial_1, color=["total_counts", "n_genes_by_counts", "leiden"], wspace=0.4)

plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata_spatial_2, color=["total_counts", "n_genes_by_counts", "leiden"], wspace=0.4)

#you can see there is large variation in total counts and the number of genes by the cell type

#plot with spatial coordinates
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_1, img_key="hires", color=["total_counts", "n_genes_by_counts"])

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata_spatial_2, img_key="hires", color=["total_counts", "n_genes_by_counts"])

#can see higher counts/genes are localized in a certain layer of the cortex, which makes sense

#plot with leiden clustering
sc.pl.spatial(adata_spatial_1, img_key="hires", color="leiden", size=1.5)
sc.pl.spatial(adata_spatial_2, img_key="hires", color="leiden", size=1.5)

#check which clusters correspond to the higher counts/genes
sc.pl.spatial(adata_spatial_1, img_key="hires", color=["leiden", "n_genes_by_counts"], size=1.5)
sc.pl.spatial(adata_spatial_2, img_key="hires", color=["leiden", "n_genes_by_counts"], size=1.5)
#and the higher counts/genes correspond to clusters 1 and 2 

#lets save the data now for later usage
adata_spatial_1.write_h5ad("D:/Major project/data/adata_spatial_1_lognorm_umap.h5ad")
adata_spatial_2.write_h5ad("D:/Major project/data/adata_spatial_2_lognorm_umap.h5ad")


#%% [9] Data Integration of multiple spatial transcriptomics 

adatas = [adata_spatial_1, adata_spatial_2]
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
    ["V1_Human_Brain_Section_1", "V1_Human_Brain_Section_2"]
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


#%% [10] DGE

sc.tl.rank_genes_groups(adata_spatial, 'leiden', method='wilcoxon')
genes_spatial_combined = extract_genes_dge(adata=adata_spatial)

genes_spatial_combined

#Lets now look at cluster 1 and 2 (corresponding to the cortex layer)
genes_spatial_combined[genes_spatial_combined.cluster=='1'] #take the top marker gene --> PCDHA4
genes_spatial_combined[genes_spatial_combined.cluster=='2'] #take the top marker gene --> MYL7

#Now lets plot the spatial map with those markers
sc.pl.spatial(adata_spatial, img_key="hires", color=["leiden", "PCDHA4", 'MYL7'], size=1.5, library_id='V1_Human_Brain_Section_1')
sc.pl.spatial(adata_spatial, img_key="hires", color=["leiden", "PCDHA4", 'MYL7'], size=1.5, library_id='V1_Human_Brain_Section_2')


#%% Tutorial 3: Data integration and label transfer from scRNA-seq dataset: cerebral cortex

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

#%% [1] Import packages 

#if you cannot run wget, use this:
import os
import pandas as pd
import requests
import scanpy as sc
import anndata as an
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

# Define function to download and decompress files
def download_and_decompress(url, filename):
    if not Path(filename).exists():
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

#Download scRNA seq data
#Used this paper's data: https://www-science-org.vu-nl.idm.oclc.org/doi/10.1126/sciadv.adg3754#acknowledgments
#Downloaded from: https://cellxgene.cziscience.com/collections/ceb895f4-ff9f-403a-b7c3-187a9657ac2c

#%% [2] Read in data 

#Read in scRNA seq data 
adata = sc.read_h5ad("D:/Major project/DLPFC_human.h5ad")

#need to run var names make unique 
adata.var_names_make_unique()

#calculate qc metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

#Quality Control

sns.histplot(adata.obs["total_counts"], kde=False)
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60)

#to zoom in more closely to the plots:
sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 2000], kde=False, bins=40)
sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] > 2000], kde=False, bins=40)
sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 2000], kde=False, bins=40)
sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] > 2000], kde=False, bins=40)

#in this case the data looks good so we wont do filtering on genes and cells. 

#to be able to calculate percentage of mitochondrial reads, we need to transform the ENSG to gene names.
#we will use scanpy to do that
#ImportError: This method requires the `pybiomart` module to be installed.: pip install pybiomart
annot = sc.queries.biomart_annotations(
    "hsapiens",
    ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"],
).set_index("ensembl_gene_id")

annot['ensembl_gene_id'] = annot.index

#substitute NaNs for the ensembl gene id
annot["external_gene_name"].fillna(annot['ensembl_gene_id'], inplace=True)


#check which of the genes in annot is present in the adata
#first make a dictionary to make ensembl gene id to gene names
ensembl_to_gene_name = annot["external_gene_name"].to_dict()

#set var and var_names
#adata.var_names = annot["external_gene_name"]
#adata.var = annot

#adata.var is 30033 
#annot is 70711

adata.var_names = [ensembl_to_gene_name.get(gene_id, gene_id) for gene_id in adata.var_names]
#adata.var = annot

#calculate % of mitochondrial reads
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

adata.obs

#check % of mitochondrial reads
sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4)

#Filtering not needed. 

#log-normalize
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

#save the adata
adata.write_h5ad("D:/Major project/data/adata_sc_lognorm.h5ad")

#read in adata
adata= sc.read_h5ad("D:/Major project/data/adata_sc_lognorm.h5ad")


##Feedback
#3D PCA/UMAP 
#
https://pair-code.github.io/understanding-umap/
https://stackoverflow.com/questions/44329068/jupyter-notebook-interactive-plot-with-widgets
#Plot marker genes in UMAP; see if clusters correspond to the cell types
#Check if the max score of the columns fits nicely 


#%% [3] Integration with spatial data

#set index to gene id
adata_spatial_1.var.set_index("gene_ids", inplace=True)
adata_spatial_2.var.set_index("gene_ids", inplace=True)

#Run integration with scanorama
adatas_1 = [adata, adata_spatial_1]
adatas_2 = [adata, adata_spatial_2]

# Integration.
adatas_cor_1 = scanorama.correct_scanpy(adatas_1, return_dimred=True)
adatas_cor_2 = scanorama.correct_scanpy(adatas_2, return_dimred=True)

#Concatenate datasets and assign integrated embeddings to anndata objects.
#Notice that we are concatenating datasets with the join="outer" and uns_merge="first" strategies. 
#This is because we want to keep the obsm['coords'] as well as the images of the visium datasets.
adata_cortex_1 = sc.concat(
    adatas_cor_1,
    label="dataset",
    keys=["scRNA-seq", "visium"],
    join="outer",
    uns_merge="first",
)
adata_cortex_2 = sc.concat(
    adatas_cor_2,
    label="dataset",
    keys=["scRNA-seq", "visium"],
    join="outer",
    uns_merge="first",
)


#At this step, we have integrated each visium dataset in a common embedding with the scRNA-seq dataset. 
#In such embedding space, we can compute distances between samples and use such distances as weights to be 
#used for for propagating labels from the scRNA-seq dataset to the Visium dataset.

#Such approach is very similar to the TransferData function in Seurat (see paper). 
#Here, we re-implement the label transfer function with a simple python function, see below.

#First, let’s compute cosine distances between the visium dataset and the scRNA-seq dataset, 
#in the common embedding space


distances_1 = 1 - cosine_distances(
    adata_cortex_1[adata_cortex_1.obs.dataset == "scRNA-seq"].obsm[
        "X_scanorama"
    ],
    adata_cortex_1[adata_cortex_1.obs.dataset == "visium"].obsm[
        "X_scanorama"
    ],
)
distances_2 = 1 - cosine_distances(
    adata_cortex_2[adata_cortex_2.obs.dataset == "scRNA-seq"].obsm[
        "X_scanorama"
    ],
    adata_cortex_2[adata_cortex_2.obs.dataset == "visium"].obsm[
        "X_scanorama"
    ],
)

#Then, let’s propagate labels from the scRNA-seq dataset to the visium dataset
def label_transfer(dist, labels):
    lab = pd.get_dummies(labels).to_numpy().T
    class_prob = lab @ dist
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    return class_prob

class_prob_1 = label_transfer(distances_1, adata.obs.cell_type)
class_prob_2 = label_transfer(
    distances_2, adata.obs.cell_type
)

#The class_prob_[anterior-posterior] objects is a numpy array of shape (cell_type, visium_spots) 
#that contains assigned weights of each spots to each cell types. 
#This value essentially tells us how similar that spots look like, from an expression profile perspective, 
#to all the other annotated cell types from the scRNA-seq dataset.

#We convert the class_prob_[anterior-posterior] object to a dataframe and assign it to the respective anndata
cp_1_df = pd.DataFrame(
    class_prob_1,
    columns=sorted(adata.obs["cell_type"].cat.categories),
)
cp_2_df = pd.DataFrame(
    class_prob_2,
    columns=sorted(adata.obs["cell_type"].cat.categories),
)

cp_1_df.index = adata_spatial_1.obs.index
cp_2_df.index = adata_spatial_2.obs.index

adata_1_transfer = adata_spatial_1.copy()
adata_1_transfer.obs = pd.concat(
    [adata_spatial_1.obs, cp_1_df], axis=1
)

adata_2_transfer = adata_spatial_2.copy()
adata_2_transfer.obs = pd.concat(
    [adata_spatial_2.obs, cp_2_df], axis=1
)

#Now we can visualize cell types 
sc.pl.spatial(
    adata_1_transfer,
    img_key="hires",
    color=['oligodendrocyte', 'astrocyte', 'glutamatergic neuron'],
    size=1.5,
)


sc.pl.spatial(
    adata_2_transfer,
    img_key="hires",
    color=['oligodendrocyte', 'astrocyte', 'glutamatergic neuron'],
    size=1.5,
)

#To zoom 
sc.pl.spatial(
    adata_2_transfer,
    img_key="hires",
    color="oligodendrocyte",
    crop_coord=[7000, 10000, 0, 6000],
    alpha=0.5,
    size=1.3,
)

# Save data
#adata.write_h5ad("D:/Major project/adata_single_cell.h5ad")
#adata_spatial_1.write_h5ad("D:/Major project/adata_spatial_1.h5ad")
#adata_spatial_2.write_h5ad("D:/Major project/adata_spatial_2.h5ad")
adata_spatial.write_h5ad("D:/Major project/data/adata_spatials_combined.h5ad")
adata_cortex_1.write_h5ad("D:/Major project/data/adata_spatial1_and_sc_combined.h5ad")
adata_cortex_2.write_h5ad("D:/Major project/data/adata_spatial2_and_sc_combined.h5ad")
adata_1_transfer.write_h5ad("D:/Major project/data/adata_spatial1_with_label.h5ad")
adata_2_transfer.write_h5ad("D:/Major project/data/adata_spatial2_with_label.h5ad")
