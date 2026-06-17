from unittest import loader

import torch
from torch.utils.data import DataLoader
import scanpy as sc

from nano_scgpt.model import scGPTForPerturbationResponsePrediction
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.perturbation_data import PerturbationDataSplitter, PerturbationDataset



if __name__ == "__main__":
    batch_size = 8
    model = scGPTForPerturbationResponsePrediction.from_pretrained("scGPT_human")

    # TODO: filter out perturbation genes in the go.cvs for GEAR like dataset?
    adata = sc.read_h5ad("../data/norman/perturb_processed.h5ad")
    adata.var['gene_symbol'] = adata.var['gene_name']

    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    tokenizer.max_length = 1536
    data_splitter = PerturbationDataSplitter(adata, tokenizer)
    train_adata, val_adata, test_adata = data_splitter.get_train_val_test()

    train_dataset = PerturbationDataset(train_adata, tokenizer, split='train')
    test_dataset = PerturbationDataset(test_adata, tokenizer, split='test')
    val_dataset = PerturbationDataset(val_adata, tokenizer, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    
    for batch in train_loader:
        gene_ids = batch["gene_ids"]
        gene_values = batch["gene_values"]
        src_key_padding_mask = batch["src_key_padding_mask"]
        pert_labels = batch["pert_labels"]
        target_values = batch["target_values"]

        pred = model(gene_ids, gene_values, src_key_padding_mask, pert_labels)
        print(pred.shape)
        break

