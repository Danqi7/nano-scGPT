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

def _preprocess_adata(adata: sc.AnnData, 
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
    if len(adata.var) > 5000:
        print("Detected more than 5000 genes. Selecting top 5000 highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)

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
        self.keep_perturbed_genes = args.keep_perturbed_genes
        self.keep_genes_per_cell = args.keep_genes_per_cell
        self.batch_size = args.batch_size
        self.amp = not args.no_amp
        self.load_splits = args.load_splits
        self.save_dir = args.save_dir
        self.logger = logger
        self.seed = args.seed

        _set_seed(self.seed)
 
        # Data Preparation.
        if not self.load_splits:
            print("Creating new train/val/test splits and saving to file.")
            self.splitter = PerturbationDataSplitter(adata, tokenizer, seed=self.seed)
            self.splitter.save_splits_to_file(f"./data/{args.data}/perturbation_splits_seed_{self.seed}.json")
        else:
            print("Loading train/val/test splits from file.")
            split_file = f"./data/{args.data}/perturbation_splits_seed_{self.seed}.json"
            self.splitter = PerturbationDataSplitter(adata, tokenizer, split_file=split_file, seed=self.seed)
        train_adata, val_adata, test_adata = self.splitter.get_train_val_test_adata()
        self.train_dataset = PerturbationDataset(train_adata, tokenizer, split='train', 
                                                 condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                                 keep_perturbed_genes=self.keep_perturbed_genes, keep_genes_per_cell=self.keep_genes_per_cell)
        self.test_dataset = PerturbationDataset(test_adata, tokenizer, split='test', 
                                                condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                                keep_perturbed_genes=self.keep_perturbed_genes, keep_genes_per_cell=self.keep_genes_per_cell)
        self.val_dataset = PerturbationDataset(val_adata, tokenizer, split='val', 
                                               condition_col=self.condition_col, condition_delimiter=self.condition_delimiter, control_condition=self.control_condition,
                                               keep_perturbed_genes=self.keep_perturbed_genes, keep_genes_per_cell=self.keep_genes_per_cell)
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

        for epoch in range(self.n_epochs):
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
            for idx, batch in enumerate(loader):
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adamson", type=str, choices=["adamson", "norman", "replogle_k562_essential"], help="Dataset to use for training and evaluation.")
    parser.add_argument("--data_file", default=None, help="Path to the data file.")
    parser.add_argument("--gene_symbol_col", default="gene_name", type=str, help="Column name in adata.var that contains gene symbols.")
    parser.add_argument("--condition_col", default="condition", type=str, help="Column name in adata.obs that contains condition names.")
    parser.add_argument("--condition_delimiter", default="+", type=str, help="Delimiter used in condition names to separate multiple genes.")
    parser.add_argument("--control_condition", default="ctrl", type=str, help="Name of the control condition in the dataset.")
    parser.add_argument("--covariate", default="cell_type", type=str, help="Covariate column in adata.obs for DGE analysis.")
    parser.add_argument("--groupby", default="condition_name", type=str, help="Column in adata.obs to group by for DGE analysis."),
    parser.add_argument("--pre_normalized", action='store_true', help="Whether the input data is already normalized. If not, normalization will be applied.")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Whether to train the model or evaluate the best saved model on the test set.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--scheduler_step_size", type=int, default=1, help="Step size for the learning rate scheduler.")
    parser.add_argument("--no_amp", action='store_true', help="Whether to disable automatic mixed precision (AMP) for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")    
    parser.add_argument("--keep_perturbed_genes", action='store_true', help="Whether to keep perturbed genes in the input data. OG scGPT default to False.")
    parser.add_argument("--keep_genes_per_cell", action='store_true', help="Whether to keep genes per cell in the input data. OG scGPT default to False.")
    parser.add_argument("--load_splits", action='store_true', help="Whether to load pre-saved train/val/test splits from a file. If not, new splits will be created and saved.")
    parser.add_argument("--log", action='store_true', help="Whether to log the training and evaluation process to a file.")

    args = parser.parse_args()

    os.makedirs(f"./results/{args.data}", exist_ok=True)
    save_dir = f"./results/{args.data}/keep_{args.keep_perturbed_genes}_keep_genes_per_cell_{args.keep_genes_per_cell}_seed_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    args.device = device
    
    logger = None
    if args.log:
        logging.basicConfig(
            filename=f"{args.save_dir}/run.log",
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        log(f"Arguments: {args}", logger)
    
    # Load data
    if not args.data_file:
        adata = sc.read_h5ad(f"./data/{args.data}/perturb_processed.h5ad")
        adata.var['gene_symbol'] = adata.var[args.gene_symbol_col]
    else:
        adata = sc.read_h5ad(args.data_file)
        adata = _preprocess_adata(adata,
                                  gene_symbol_col=args.gene_symbol_col, 
                                  covariate=args.covariate, 
                                  condition_col=args.condition_col,
                                  condition_delimiter=args.condition_delimiter,
                                  control_condition=args.control_condition, 
                                  groupby=args.groupby, 
                                  pre_normalized=args.pre_normalized)

    
    # Tokenizer
    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    tokenizer.max_length = 1536

    # Model
    model = scGPTForPerturbationResponsePrediction.from_pretrained("scGPT_human")

    trainer = PerturbationTrainer(model=model, adata=adata, tokenizer=tokenizer, args=args, logger=logger)
    if args.mode == 'train':
        trainer.train()
    
    # Evaluate best model on test set.
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=args.device))
    metrics, metrics_by_perts, subgroup_metrics = trainer.evaluate(model, 
                                                                   trainer.test_loader, 
                                                                   device=args.device, 
                                                                   amp=not args.no_amp, 
                                                                   logger=logger, 
                                                                   subgroups=trainer.test_subgroups)
    if args.save_dir:
        with open(f"{args.save_dir}/eval_metrics.json", "w") as f:
            json.dump({
                "metrics": metrics,
                "metrics_by_perts": metrics_by_perts,
                "subgroup_metrics": subgroup_metrics
            }, f, indent=4)

    print("Training and evaluation completed.")