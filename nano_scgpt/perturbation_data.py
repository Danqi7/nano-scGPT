import numpy as np
import torch
from torch.utils.data import Dataset

class PerturbationDataset(Dataset):

    def __init__(self, adata, tokenizer, num_ctrl=1):
        self.adata = adata
        self.tokenizer = tokenizer # NOTE: need to flag `filter_zero_expr_genes=False` for the perturbation task since we want to keep the zero-expression genes for prediction.
        self.gene_names = adata.var['gene_symbol'].tolist()

        self.vocab_genes_idx = [idx for idx, g in enumerate(self.gene_names) if g in self.tokenizer.vocab]
        self.aligned_gene_ids = np.array([self.tokenizer.vocab[self.gene_names[idx]] for idx in self.vocab_genes_idx]) # shape [G_vocab]
        if len(self.aligned_gene_ids) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        
        self.X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        
        print(f"Original genes: {len(self.gene_names)}| Genes in vocab: {len(self.aligned_gene_ids)}")

        # fixed gene set for all cells.
        self.gene_ids = np.array([
            self.tokenizer.vocab.get(g, 0)   # OOV → 0 (pad)
            for g in self.gene_names
        ])
        
        self.T = self.tokenizer.max_length
        self.ctrl_idx = np.where(adata.obs['condition'] == 'ctrl')[0]

        self.pairs = [] # [(ctrl_idx, pert_idx, pert_names)]
        for idx, row in enumerate(adata.obs.itertuples()):
            condition = row.condition
            if condition != 'ctrl':
                pert_genes = [g for g in condition.split("+") if g != "ctrl"]
                sampled_ctrl_idx = self.ctrl_idx[np.random.randint(0, len(self.ctrl_idx), num_ctrl)]
                for c_idx in sampled_ctrl_idx:
                    self.pairs.append((c_idx, idx, pert_genes))
            else:
                self.pairs.append((idx, idx, ['ctrl']))
        
        # TODO: Do DGE analysis for each perturbation vs ctrl and save the top K DE genes for evaluation.
        self.perturbations = adata.obs['condition'].unique().tolist()
        
    def __len__(self):
        return len(self.pairs)
    
    def __str__(self):
        return super().__str__() + f"| num_pairs: {len(self.pairs)} | num perturbations: {len(self.perturbations)}"
    
    def __getitem__(self, index):
        ctrl_idx, pert_idx, pert_names = self.pairs[index]

        # full gene set
        ctrl_exprs = self.X[ctrl_idx]            # (n_genes,)
        pert_exprs = self.X[pert_idx]            # (n_genes,)


        # pert flags over the full fixed gene set
        pert_labels = np.isin(self.gene_names, pert_names).astype(np.int64)  # (n_genes,)

        return {
            "gene_values":          torch.from_numpy(ctrl_exprs).float(),
            "pert_labels":          torch.from_numpy(pert_labels).long(),
            "target_values":        torch.from_numpy(pert_exprs).float(),
        }
    
    def collate_fn(self, batch):
        gene_values  = torch.stack([item["gene_values"]   for item in batch])  # (B, n_genes)
        pert_labels  = torch.stack([item["pert_labels"]   for item in batch])
        target_values= torch.stack([item["target_values"] for item in batch])
        B, n_genes   = gene_values.shape

        # sample gene subset ONCE for the whole batch
        if n_genes > self.T:
            # ![TODO][NOTE]: this means perturbed genes may get dropped with probability (T/n_genes), which flags the whole pair as NOT perturbed but in reality it is.
            # This can be misleading for the model, and degrade fine-tuning performance on the perturbation prediction task. 
            # A potential solution is to always keep the perturbed genes and only sample from the non-perturbed genes to fill up the T tokens.
            idx = torch.randperm(n_genes)[:self.T]
            gene_values   = gene_values[:, idx]
            pert_labels   = pert_labels[:, idx]
            target_values = target_values[:, idx]
            gene_ids      = torch.from_numpy(self.gene_ids[idx]).long().unsqueeze(0).repeat(B, 1)
        else:
            gene_ids      = torch.from_numpy(self.gene_ids).long().unsqueeze(0).repeat(B, 1)

        # all-False padding mask
        src_key_padding_mask = torch.zeros_like(gene_values, dtype=torch.bool)

        return {
            "gene_ids":             gene_ids,
            "gene_values":          gene_values,
            "src_key_padding_mask": src_key_padding_mask,
            "pert_labels":          pert_labels,
            "target_values":        target_values,
        }
