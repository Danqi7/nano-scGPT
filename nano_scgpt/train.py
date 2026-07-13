import torch
from torch.utils.data import DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist

from nano_scgpt.model import scGPTForPerturbationResponsePrediction
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer, _check_log1ped
from nano_scgpt.perturbation_data import PerturbationDataSplitter, PerturbationDataset
from nano_scgpt.eval_metrics import compute_perturbation_metrics

from typing import Dict, List, Tuple
import os
import json
import random
import warnings
import argparse
import logging
from tqdm import tqdm

def _set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)

def preprocess_adata(adata: sc.AnnData, 
                      gene_symbol_col : str, 
                      covariate: str, 
                      condition_col: str,
                      condition_delimiter: str,
                      control_condition: str, 
                      groupby: str, 
                      pre_normalized: bool) -> sc.AnnData:
    assert gene_symbol_col in adata.var.columns, f"Column '{gene_symbol_col}' not found in adata.var. Please provide a valid column name that contains gene symbols."
    assert condition_col in adata.obs.columns, f"Column '{condition_col}' not found in adata.obs. Please provide a valid column name that contains condition names."

    adata.var['gene_symbol'] = adata.var[gene_symbol_col]

    if not pre_normalized:
        print("Normalizing the data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
    if not _check_log1ped(adata.X):
        print("Applying log1p transformation to the data...")
        sc.pp.log1p(adata)
    if len(adata.var) > 6000:
        print("Detected more than 6000 genes. Selecting max top 6000 highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=6000, subset=True)

    if 'rank_genes_groups_cov_all' not in adata.uns:
        print("Computing rank_genes_groups_cov_all for later evaluation ...")
        adata.obs.loc[:, 'dose_val'] = adata.obs[condition_col].apply(lambda x: '1+1' if len(x.split(condition_delimiter)) == 2 else '1') # ctrl -> 1, perturbation -> 1+1.
        adata.obs.loc[:, 'condition_name'] =  adata.obs.apply(lambda x: '_'.join([x[covariate], x[condition_col] , x.dose_val]), axis = 1) # {covariate}_{condition}_{dose_val}
    
        control_group = control_condition + "_" + "1" # e.g. ctrl_1
        gene_dict = {} # {condition_name: [list of DE genes]}

        cov_categories = adata.obs[covariate].unique()
        for cov_cat in cov_categories:
            control_group_cov = '_'.join([cov_cat, control_group]) # e.g. {cov_cat}_ctrl_1

            adata_cov = adata[adata.obs[covariate] == cov_cat]

            sc.tl.rank_genes_groups(
                adata_cov,
                groupby=groupby,
                reference=control_group_cov,
                rankby_abs=True,
                n_genes=len(adata.var),
                use_raw=False
            )

            de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
            for group in de_genes:
                gene_dict[group] = de_genes[group].tolist()

        adata.uns["rank_genes_groups_cov_all"] = gene_dict

    return adata

class PerturbationTrainer:
    def __init__(self, model, adata, tokenizer, args: Dict, logger=None):
        self.model = model.to(args.device)
        self.adata = adata
        self.tokenizer = tokenizer

        self.condition_col = args.condition_col
        self.condition_delimiter = args.condition_delimiter
        self.control_condition = args.control_condition

        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.scheduler_step_size = args.scheduler_step_size
        self.early_stopping_patience = args.early_stopping_patience
        self.device = args.device
        self.keep_genes_per_cell = args.keep_genes_per_cell
        self.batch_size = args.batch_size
        self.amp = not args.no_amp
        self.load_splits = args.load_splits
        self.save_dir = args.save_dir
        self.logger = logger
        self.seed = args.seed

        _set_seed(self.seed)
 
        # Data Preparation.
        split_file = f"./data/{args.data}/perturbation_splits_seed_{self.seed}.json"
        if not self.load_splits or not os.path.exists(split_file):
            print("Creating new train/val/test splits and saving to file.")
            self.splitter = PerturbationDataSplitter(adata, tokenizer, seed=self.seed)
            self.splitter.save_splits_to_file(split_file)
        else:
            print("Loading train/val/test splits from file.")
            self.splitter = PerturbationDataSplitter(adata, tokenizer, split_file=split_file, seed=self.seed)
        train_adata, val_adata, test_adata = self.splitter.get_train_val_test_adata()
        self.train_dataset = PerturbationDataset(train_adata, tokenizer, split='train', 
                                                 condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                                 keep_genes_per_cell=self.keep_genes_per_cell)
        self.test_dataset = PerturbationDataset(test_adata, tokenizer, split='test', 
                                                condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                                keep_genes_per_cell=self.keep_genes_per_cell)
        self.val_dataset = PerturbationDataset(val_adata, tokenizer, split='val', 
                                               condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                               keep_genes_per_cell=self.keep_genes_per_cell)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.test_dataset.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn)
        self.test_subgroups = self.splitter.test_subgroups

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size, gamma=0.9)
        self.scaler    = torch.amp.GradScaler(enabled=self.amp)

        self.best_val_metric = -float('inf')

    
    def train(self):
        patience = 0

        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs", unit="epoch"):
            self.model.train()
            train_loss = 0.0

            for idx, batch in enumerate(self.train_loader):
                gene_ids = batch["gene_ids"].to(self.device)
                gene_values = batch["gene_values"].to(self.device)
                src_key_padding_mask = batch["src_key_padding_mask"].to(self.device)
                pert_labels = batch["pert_labels"].to(self.device)
                target_values = batch["target_values"].to(self.device)

                with torch.amp.autocast(device_type=self.device, enabled=self.amp):
                    pred = self.model(gene_ids, gene_values, src_key_padding_mask, pert_labels)
                    loss = torch.nn.functional.mse_loss(pred, target_values, reduction='mean')

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer) # has to explicitly unscale before clipping gradients.
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        1.0,
                        error_if_nonfinite=False if self.scaler.is_enabled() else True,
                    )
                    if len(w) > 0:
                        print(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                if epoch == 0 or epoch > 0 and idx % 100 == 0:
                    log(f"Epoch {epoch+1}/{self.n_epochs}, Step {idx+1} | Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.4f} | Scaler: {self.scaler.get_scale()} | Norm: {norm:.4f}", self.logger)

            self.scheduler.step()
    
            # Evaluation.
            metrics, _, _ = self.evaluate(self.model, self.val_loader, device=self.device, amp=self.amp, logger=self.logger)

            # Early stopping based on pearson correlation across genes.
            # !NOTE: can also use pearson delta instead.
            metric_name = "pearson"
            if metrics[metric_name] > self.best_val_metric:
                self.best_val_metric = metrics[metric_name]
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")
                log(f"New best model saved with {metric_name}: {self.best_val_metric:.4f}", self.logger)
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    log(f"Early stopping triggered after {epoch+1} epochs.", self.logger)
                    break
    
    def evaluate(self, model, loader, device, amp=True, logger=None, subgroups=None):
        model.eval()

        ctrl_adata = loader.dataset.adata[loader.dataset.adata.obs['condition'] == 'ctrl']
        predictions = []
        gts = []
        perts = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating", unit="batch"):
                gene_ids = batch["gene_ids"].to(device)
                gene_values = batch["gene_values"].to(device)
                src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                pert_labels = batch["pert_labels"].to(device)
                target_values = batch["target_values"].to(device)

                with torch.amp.autocast(device_type=device, enabled=amp):
                    pred = model(gene_ids, gene_values, src_key_padding_mask, pert_labels) # (B, n_genes)
                    
                    predictions.extend(pred.detach().cpu().numpy())
                    gts.extend(target_values.detach().cpu().numpy())
                    perts.extend(batch["perturbations"])

        predictions = np.stack(predictions)
        gts = np.stack(gts)
        metrics, metrics_by_perts, subgroup_metrics = compute_perturbation_metrics(
            predictions, 
            gts, 
            np.array(perts), 
            ctrl_adata,
            subgroups=subgroups,
            control_condition=self.control_condition
        )
        log(f"Test Prediction Shape: {predictions.shape}, GT Shape: {gts.shape}, Perturbations Length: {len(perts)}", logger)
        log(f"Eval | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()), logger)
        log(f"Metrics by Perturbations: {metrics_by_perts}", logger)
        if subgroup_metrics:
            log(f"Subgroup Metrics: {subgroup_metrics}", logger)
    
        return metrics, metrics_by_perts, subgroup_metrics
