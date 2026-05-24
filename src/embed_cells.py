"""
Embed cells using scGPT
"""
import numpy as np
import scgpt
import scgpt.preprocess as scgpt_preprocess
import scanpy as sc
import pickle

def embed_cells(adata, model_dir='../pretrained_weights/scGPT_human/'):
    data_is_raw = True
    n_bins = 51
    # Preprocess the cells s.t. remove low quality cells/genes; normalize raw count data; build gene vocab and bin expressions.
    preprocessor = scgpt_preprocess.Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=False,  # 6. whether to bin the raw data and to what number of bins NOTE: double check cuz data collator binnied it again.
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    preprocessor(adata, batch_key=None)

    # Load pretrained model and tokenizer

    # Tokenize the cells

    # Embed the cells
    gene_col = "gene_symbol"
    cell_type_key = None
    ref_embed_adata = scgpt.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        obs_to_save=cell_type_key,  # optional arg, only for saving metainfo
        batch_size=64,
        return_new_adata=True,
    )


    # Save the embeddings

    return ref_embed_adata, ref_embed_adata.X


if __name__ == "__main__":
    adata = sc.read_h5ad('../data/lung.h5ad')
    # Read the gene symbol list from the pickle file
    # TODO: some ensembl ids mapped to nan. 
    with open('../data/gene_symbols.pkl', 'rb') as f:
        gene_symbols = pickle.load(f)
    adata.var['gene_symbol'] = gene_symbols
    # Filter out gene_symbols nan
    adata = adata[ :, ~adata.var['gene_symbol'].isna() ]
    print(f"Embedding adata of shape {adata.shape} ...")

    _, og_embeddings = embed_cells(adata)

    # save the embeddings
    np.save("../outputs/ref_scGPT_embds.npy", og_embeddings)

