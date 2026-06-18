from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
    

class PerturbationDataSplitter:

    def __init__(self, adata, tokenizer, 
                 train_gene_set_size=0.75,
                 combo_seen2_train_size=0.75,
                 train_val_gene_set_size=0.9,
                 train_val_combo_seen2_train_size=0.9,
                 splits=['train', 'val', 'test'], 
                 seed=1):
        self.adata = adata
        self.tokenizer = tokenizer
        self.splits = splits

        pert_names = adata.obs['condition'].unique().tolist()
        pert_names.remove("ctrl")
        assert splits == ['train', 'val', 'test'], "Only support train/val/test splits for now."
        
        train_pert_names, test_pert_names, test_subgroups = self.split_perturbations(pert_names,
                                                                        train_gene_set_size=train_gene_set_size, 
                                                                        combo_seen2_train_size=combo_seen2_train_size,
                                                                        seed=seed)
        train_pert_names, val_pert_names, val_subgroups = self.split_perturbations(train_pert_names,
                                                                        train_gene_set_size=train_val_gene_set_size, 
                                                                        combo_seen2_train_size=train_val_combo_seen2_train_size,
                                                                        seed=seed)
        self.train_pert_names = train_pert_names
        self.val_pert_names = val_pert_names
        self.test_pert_names = test_pert_names
        self.test_subgroups = test_subgroups
        self.val_subgroups = val_subgroups

        print('Test subgroups:')
        for k, v in test_subgroups.items():
            print(f"{k}: {len(v)} perturbations")

        # Map perturbation names to splits and create adata subsets for each split.
        pert2split = {}
        for p in train_pert_names:
            pert2split[p] = "train"
        for p in val_pert_names:
            pert2split[p] = "val"
        for p in test_pert_names:
            pert2split[p] = "test"
        pert2split["ctrl"] = "ctrl"
        adata.obs['split'] = adata.obs['condition'].map(pert2split)
        
        self.train_adata = adata[(adata.obs['split']  == "train") | (adata.obs['split'] == "ctrl")]
        self.test_adata = adata[(adata.obs['split'] == "test") | (adata.obs['split'] == "ctrl")]
        self.val_adata = adata[(adata.obs['split'] == "val") | (adata.obs['split'] == "ctrl")]

    def get_train_val_test(self):
        return self.train_adata, self.val_adata, self.test_adata

    def get_gene_set_from_perturnations(self, pert_names: List[str]) -> List[str]:
        """Extract the unique set of genes in the perturbations from the list of perturbation names, excluding the "ctrl" condition."""
        gene_set = set()
        for pert in pert_names:
            genes = pert.split("+")
            for g in genes:
                if g != "ctrl":
                    gene_set.add(g)
        return sorted(list(gene_set))

    def split_perturbations(self, pert_names, 
                            train_gene_set_size=0.75, 
                            combo_seen2_train_size=0.75, 
                            seed=1) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Split perturbations into train and test sets, and further categorize the test perturbations into combo_seen0, combo_seen1, combo_seen2, 
        and single_unseen based on the presence of their individual genes in the training set.

        Args:
            pert_names: list of perturbation names, e.g. ["geneA", "geneB", "geneA+ctrl", ...], excluding "ctrl".
            train_gene_set_size: fraction of individual genes to be included in the training set.
            combo_seen2_train_size: fraction of comb perturbations with both genes individually seen in the train set to be included in the train set.
            seed: random seed for reproducibility.
        Returns:
            train_pert_names: list of perturbation names for training.
            test_pert_names: list of perturbation names for testing.
            test_subgroups: dict with keys "combo_seen0", "combo_seen1", "combo_seen2", "single_unseen", each containing a list of perturbation names for that category.
        """
        np.random.seed(seed)

        train_names = []
        test_names = []
        unique_genes = self.get_gene_set_from_perturnations(pert_names)

        train_genes_candidates = np.random.choice(unique_genes, size=int(len(unique_genes)*train_gene_set_size), replace=False)
        ood_genes = np.setdiff1d(unique_genes, train_genes_candidates)
        print(f"Split perturbations - Unique perturbed genes: {len(unique_genes)} | Train gene candidates: {len(train_genes_candidates)} | OOD genes: {len(ood_genes)}")

        # All single-gene perturbations with genes in the train candidates go to the train set;
        # combo-gene perturbations with 1 gene in the train candidates go to the test set as combo_seen1;
        # frac of combo-gene perturbations with both genes in the train candidates go to the train set as combo_seen2, the rest go to the test set as combo_seen2;
        # combo-gene perturbations with both genes in the ood genes go to the test set as combo_seen0;
        # single-gene perturbations with genes in the ood genes go to the test set as single_unseen.
        test_subgroups = {"combo_seen0": [], "combo_seen1": [], "combo_seen2": [], "single_unseen": []}
        combo2_candidates = []
        for pert in pert_names:
            genes = pert.split("+")
            if len(genes) == 1 or genes[0] == "ctrl" or genes[1] == "ctrl": # single-gene perturbation
                current_gene = genes[0] if genes[0] != "ctrl" else genes[1]
                if current_gene in train_genes_candidates:
                    train_names.append(pert)
                else:
                    test_names.append(pert)
                    test_subgroups["single_unseen"].append(pert)
            else: # combo-gene perturbation
                num_train_genes = sum([1 if g in train_genes_candidates else 0 for g in genes])
                if num_train_genes == 2:
                    combo2_candidates.append(pert)
                elif num_train_genes == 1:
                    test_names.append(pert)
                    test_subgroups["combo_seen1"].append(pert)
                else:
                    test_names.append(pert)
                    test_subgroups["combo_seen0"].append(pert)
        
        train_combo2 = np.random.choice(combo2_candidates, size=int(len(combo2_candidates)*combo_seen2_train_size), replace=False)
        train_names.extend(train_combo2)

        test_combo2 = np.setdiff1d(combo2_candidates, train_combo2)
        test_names.extend(test_combo2)
        test_subgroups["combo_seen2"].extend(test_combo2)

        assert len(train_names) + len(test_names) == len(pert_names)
        assert len(test_subgroups["combo_seen0"]) + len(test_subgroups["combo_seen1"]) + len(test_subgroups["combo_seen2"]) + len(test_subgroups["single_unseen"]) == len(test_names)

        return train_names, test_names, test_subgroups


class PerturbationDataset(Dataset):

    def __init__(self, adata, tokenizer, split='train', num_ctrl=1):
        self.adata = adata
        self.tokenizer = tokenizer # NOTE: need to flag `filter_zero_expr_genes=False` for the perturbation task since we want to keep the zero-expression genes for prediction.
        self.gene_names = adata.var['gene_symbol'].tolist()
        self.split = split

        self.vocab_genes_idx = [idx for idx, g in enumerate(self.gene_names) if g in self.tokenizer.vocab]
        self.aligned_gene_ids = np.array([self.tokenizer.vocab[self.gene_names[idx]] for idx in self.vocab_genes_idx]) # shape [G_vocab]
        if len(self.aligned_gene_ids) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        
        self.X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        
        print(f"PerturbationDataset - Original genes: {len(self.gene_names)}| Genes in vocab: {len(self.aligned_gene_ids)}")

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
                    self.pairs.append((c_idx, idx, pert_genes, condition))
            elif split == 'train': # only include ctrl-ctrl pairs in the training set.
                self.pairs.append((idx, idx, ['ctrl'], condition))
        
        # TODO: Do DGE analysis for each perturbation vs ctrl and save the top K DE genes for evaluation.
        self.perturbations = adata.obs['condition'].unique().tolist()
        
    def __len__(self):
        return len(self.pairs)
    
    def __str__(self):
        return super().__str__() + f"| num_pairs: {len(self.pairs)} | num perturbations: {len(self.perturbations)}"
    
    def __getitem__(self, index):
        ctrl_idx, pert_idx, pert_names, condition = self.pairs[index]

        # full gene set
        ctrl_exprs = self.X[ctrl_idx]            # (n_genes,)
        pert_exprs = self.X[pert_idx]            # (n_genes,)


        # pert flags over the full fixed gene set
        pert_labels = np.isin(self.gene_names, pert_names).astype(np.int64)  # (n_genes,)

        return {
            "gene_values":          torch.from_numpy(ctrl_exprs).float(),
            "pert_labels":          torch.from_numpy(pert_labels).long(),
            "target_values":        torch.from_numpy(pert_exprs).float(),
            "perturbation":         condition,
        }
    
    def collate_fn(self, batch):
        gene_values  = torch.stack([item["gene_values"]   for item in batch])  # (B, n_genes)
        pert_labels  = torch.stack([item["pert_labels"]   for item in batch])
        target_values= torch.stack([item["target_values"] for item in batch])
        perturbations = [item["perturbation"] for item in batch] # list of perturbation names in the batch, e.g. ["geneA+ctrl", "geneB+geneC", ...]
        B, n_genes   = gene_values.shape

        # sample gene subset ONCE for the whole batch, only for train. for val/test, full gene sets are returned.
        if self.split == 'train' and n_genes > self.T:
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
            "perturbations":        perturbations,
        }
