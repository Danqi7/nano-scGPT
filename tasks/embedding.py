import os
import scanpy as sc

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from nano_scgpt.scGPT_tokenizer import _check_log1ped, scGPTTokenizer, scGPTDataset
from nano_scgpt.model import scGPTModel

DEFAULT_INPUT_URL = "https://datasets.cellxgene.cziscience.com/d6761a21-e226-434f-9370-fbcc7e549aa0.h5ad"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the local input .h5ad file.")
    parser.add_argument("--input_url", type=str, required=False, default=DEFAULT_INPUT_URL, help="URL to the input .h5ad file. Ignored if --input is provided.")
    parser.add_argument("--output", type=str, default="scGPT_embeddings.npy", help="Path to save the output embeddings.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing the data.")

    args = parser.parse_args()

    # Load the input data.
    if args.input:
        adata = sc.read(args.input)
    else:
        print(f"Downloading input data from {args.input_url}...")
        adata = sc.read("data/tmp.h5ad", backup_url=args.input_url)

    # Ensure gene symbols are available in adata.var['gene_symbol'].
    if 'gene_symbol' not in adata.var.columns:
         print("The input .h5ad file has no 'gene_symbol' column, checking if the var_names contains gene symbols...")
         var_names = adata.var_names.astype(str)

         if all([s.startswith("ENSG") for s in var_names[:100]]):
            print("Detected Ensembl IDs in var_names. Checking column `feature_name` for gene symbols...")
            if 'feature_name' in adata.var.columns:
                adata.var['gene_symbol'] = adata.var['feature_name']
            else:
                raise ValueError("No gene symbols found in the input data. Please provide a .h5ad file with gene symbols in the 'gene_symbol' or 'feature_name' column or as var_names.")
         else:
            print("var_names appears to contain gene symbols. Adding to 'gene_symbol' column...")
            adata.var['gene_symbol'] = var_names
    adata.var_names = adata.var["gene_symbol"].astype(str)
    adata.var_names_make_unique(join="_")
    print(f"Embedding adata of shape {adata.shape}...")
    
    # Normalize the data.
    print("Normalizing the data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    if not _check_log1ped(adata.X):
        sc.pp.log1p(adata)

    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    dataset = scGPTDataset(adata, tokenizer)
    try:
        num_workers = min(len(os.sched_getaffinity(0)) - 1, args.batch_size)
    except AttributeError:
        num_workers = min(os.cpu_count(), 0, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Load Model.
    model = scGPTModel.from_pretrained("scGPT_human")
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    embeddings = []
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda"):
        for batch in tqdm(dataloader, desc="Embedding cells"):
            gene_ids, exprs, padding_mask = batch["gene_ids"], batch["exprs"], batch["padding_mask"]
            gene_ids, exprs, padding_mask = gene_ids.to(device), exprs.to(device), padding_mask.to(device)

            batch_embeddings = model.encode(gene_ids, exprs, padding_mask) # shape [B, D]

            embeddings.append(batch_embeddings.cpu())


    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)

    # Save the embeddings to the output path
    np.save(args.output, embeddings.numpy())
    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to {args.output}")