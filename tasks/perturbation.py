import torch
from torch.utils.data import DataLoader
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist

from nano_scgpt.model import scGPTForPerturbationResponsePrediction
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.perturbation_data import PerturbationDataSplitter, PerturbationDataset

import os
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

def compute_perturbation_metrics(preds: np.ndarray, gts: np.ndarray, pert_names:np.ndarray, ctrl_adata: sc.AnnData):
    """
    Args:
        preds (`np.ndarray` of shape `[n_cells, n_genes]`):
            predicted expression under perturbation.
        gts (`np.ndarray` of shape `[n_cells, n_genes]`):
            ground truth expression under perturbation.
        pert_names (`np.ndarray` of shape `[n_cells]`):
            array of perturbation names corresponding to each cell. e.g ["geneA+ctrl", "geneB+geneC", "geneA+ctrl", ...]
        ctrl_adata (`sc.AnnData`):
            AnnData object only containing control cells.
    Returns:
        metrics_across_genes: dict of lists, where each list contains the metric values for each gene, averaged across perturbations.
    """
    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    } # across gene metrics, averaged across perturbations.

    unique_pert_names = np.unique(pert_names)
    assert not 'ctrl' in unique_pert_names, "Control condition should not be included in the perturbation names."
    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten() # (n_genes,)
    assert mean_ctrl.max() < 1e3, "Mean control expression seems too high, make sure it is log-normalized and not raw counts."

    pert2idx = {pert: np.where(pert_names == pert)[0] for pert in unique_pert_names}
    preds_by_pert = np.array([preds[pert2idx[pert]].mean(axis=0) for pert in unique_pert_names]) # [n_perts, n_genes], avg gene exprs for each perturbation.
    gts_by_pert = np.array([gts[pert2idx[pert]].mean(axis=0) for pert in unique_pert_names]) # [n_perts, n_genes]
    preds_delta_by_pert = preds_by_pert - mean_ctrl # [n_perts, n_genes]
    gts_delta_by_pert = gts_by_pert - mean_ctrl # [n_perts, n_genes]

    # 1. Pearson correlation across genes, averaged across perturbations.
    zero_rows = np.all(gts_by_pert == 0, axis=1)   # bool mask for rows where all gene expressions are zero
    valid = ~zero_rows
    for x, y in zip(preds_by_pert[valid], gts_by_pert[valid]):
        metrics_across_genes["pearson"].append(pearsonr(x, y)[0]) # [n_perts]
    
    # 2. Pearson correlation across genes for delta expression, averaged across perturbations.
    for x, y in zip(preds_delta_by_pert[valid], gts_delta_by_pert[valid]):
        metrics_across_genes["pearson_delta"].append(pearsonr(x, y)[0]) # [n_perts]
    
    # Differential expression (DE) gene.
    top_n = 20
    gene2idx = {gene: idx for idx, gene in enumerate(ctrl_adata.var.index.values)}
    de_genes_by_pert = []
    de_genes_idx_by_pert = []
    for pert in unique_pert_names:
        key_components= next(iter(ctrl_adata.uns["rank_genes_groups_cov_all"].keys())).split("_")
        condition_key = "_".join([key_components[0], pert, key_components[2]])
        de_genes = ctrl_adata.uns["rank_genes_groups_cov_all"][condition_key]
        de_genes = de_genes[:top_n]
        de_genes_by_pert.append(de_genes)
        de_genes_idx_by_pert.append([gene2idx[gene] for gene in de_genes])
    mean_ctrl_de = np.array([mean_ctrl[idxs] for idxs in de_genes_idx_by_pert]) # [n_perts, top_n]
    preds_by_pert_de = np.array([preds_by_pert[i, idxs] for i, idxs in enumerate(de_genes_idx_by_pert)]) # [n_perts, top_n]
    gts_by_pert_de = np.array([gts_by_pert[i, idxs] for i, idxs in enumerate(de_genes_idx_by_pert)]) # [n_perts, top_n]
    preds_delta_by_pert_de = preds_by_pert_de - mean_ctrl_de # [n_perts, top_n]
    gts_delta_by_pert_de = gts_by_pert_de - mean_ctrl_de # [n_perts, top_n]

    # 3. Pearson correlation across DE genes.
    zero_rows_de = np.all(gts_by_pert_de == 0, axis=1)   # bool mask for rows where all DE gene expressions are zero
    valid_de = ~zero_rows_de
    for x, y in zip(preds_by_pert_de[valid_de], gts_by_pert_de[valid_de]):
        metrics_across_genes["pearson_de"].append(pearsonr(x, y)[0]) # [n_perts]
    
    # 4. Pearson correlation across DE genes for delta expression.
    for x, y in zip(preds_delta_by_pert_de[valid_de], gts_delta_by_pert_de[valid_de]):
        metrics_across_genes["pearson_de_delta"].append(pearsonr(x, y)[0]) # [n_perts]

    # 5. Compute correlation on ground truth perturbation distances vs predicted perturbation distances.
    # Compute pairwise distances between perturbations based on their mean expression profiles.
    gt_distances = pdist(gts_by_pert, metric='euclidean') # [n_perts * (n_perts - 1) / 2]
    pred_distances = pdist(preds_by_pert, metric='euclidean') # [n_perts * (n_perts - 1) / 2]
    # Flatten the distance matrices and compute Pearson correlation between them.
    metrics_across_genes["pearson_perturbation_distance"] = pearsonr(gt_distances, pred_distances)[0]
    metrics_across_genes["spearman_perturbation_distance"] = spearmanr(gt_distances, pred_distances)[0]


    metrics_across_genes = {k: np.mean(v) for k, v in metrics_across_genes.items()}

    return metrics_across_genes



def train(model, train_loader, val_loader, n_epochs=15, lr=1e-4, step_size=1, device='cuda', amp=True, early_stopping_patience=10, save_dir="./", logger=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
    scaler    = torch.amp.GradScaler(enabled=amp)

    patience = 0
    best_val_metric = -float('inf')
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for idx, batch in enumerate(train_loader):
            gene_ids = batch["gene_ids"].to(device)
            gene_values = batch["gene_values"].to(device)
            src_key_padding_mask = batch["src_key_padding_mask"].to(device)
            pert_labels = batch["pert_labels"].to(device)
            target_values = batch["target_values"].to(device)

            with torch.amp.autocast(device_type=device, enabled=amp):
                pred = model(gene_ids, gene_values, src_key_padding_mask, pert_labels)
                loss = torch.nn.functional.mse_loss(pred, target_values, reduction='mean')

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # has to explicitly unscale before clipping gradients.
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    print(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if epoch == 0 or epoch > 0 and idx % 100 == 0:
                log(f"Epoch {epoch+1}/{n_epochs}, Step {idx+1} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.4f} | Scaler: {scaler.get_scale()} | Norm: {norm:.4f}", logger)

        scheduler.step()
    
        # Evaluation.
        metrics = evaluate(model, val_loader, device=device, amp=amp, logger=logger)

        # Early stopping based on pearson correlation across genes.
        # !NOTE: can also use pearson delta instead.
        metric_name = "pearson"
        if metrics[metric_name] > best_val_metric:
            best_val_metric = metrics[metric_name]
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            log(f"New best model saved with {metric_name}: {best_val_metric:.4f}", logger)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                log(f"Early stopping triggered after {epoch+1} epochs.", logger)
                break
    
def evaluate(model, test_loader, device='cuda', amp=True, logger=None):
    model.eval()

    ctrl_adata = test_loader.dataset.adata[test_loader.dataset.adata.obs['condition'] == 'ctrl']
    predictions = []
    gts = []
    perts = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
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
        metrics = compute_perturbation_metrics(predictions, gts, np.array(perts), ctrl_adata)
        log(f"Test Prediction Shape: {predictions.shape}, GT Shape: {gts.shape}, Perturbations Length: {len(perts)}", logger)
        log(f"Eval | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()), logger)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Whether to train the model or evaluate the best saved model on the test set.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--scheduler_step_size", type=int, default=1, help="Step size for the learning rate scheduler.")
    parser.add_argument("--no_amp", action='store_true', help="Whether to disable automatic mixed precision (AMP) for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data", default="adamson", type=str, help="Dataset to use for training and evaluation.")
    parser.add_argument("--keep_perturbed_genes", action='store_true', help="Whether to keep perturbed genes in the input data. OG scGPT default to False.")
    parser.add_argument("--keep_genes_per_cell", action='store_true', help="Whether to keep genes per cell in the input data. OG scGPT default to False.")
    parser.add_argument("--log", action='store_true', help="Whether to log the training and evaluation process to a file.")
    args = parser.parse_args()

    _set_seed(args.seed)

    batch_size = args.batch_size
    model = scGPTForPerturbationResponsePrediction.from_pretrained("scGPT_human")

    # TODO: filter out perturbation genes in the go.cvs for GEAR like dataset?
    adata = sc.read_h5ad(f"./data/{args.data}/perturb_processed.h5ad")
    adata.var['gene_symbol'] = adata.var['gene_name']

    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    tokenizer.max_length = 1536
    data_splitter = PerturbationDataSplitter(adata, tokenizer, seed=args.seed)
    train_adata, val_adata, test_adata = data_splitter.get_train_val_test()

    train_dataset = PerturbationDataset(train_adata, tokenizer, split='train', keep_perturbed_genes=args.keep_perturbed_genes, keep_genes_per_cell=args.keep_genes_per_cell)
    test_dataset = PerturbationDataset(test_adata, tokenizer, split='test', keep_perturbed_genes=args.keep_perturbed_genes, keep_genes_per_cell=args.keep_genes_per_cell)
    val_dataset = PerturbationDataset(val_adata, tokenizer, split='val', keep_perturbed_genes=args.keep_perturbed_genes, keep_genes_per_cell=args.keep_genes_per_cell)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    
    model = model.to(device)

    os.makedirs(f"./results/{args.data}", exist_ok=True)
    save_dir = f"./results/{args.data}/keep_{args.keep_perturbed_genes}_keep_genes_per_cell_{args.keep_genes_per_cell}_seed_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    logger = None
    if args.log:
        logging.basicConfig(
            filename=f"{save_dir}/run.log",
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        log(f"Arguments: {args}", logger)
    
    # Log the perturbations in train, val, and test sets.
    log(f"Train Perturbations: {sorted(data_splitter.train_pert_names)}", logger)
    log(f"Validation Perturbations: {sorted(data_splitter.val_pert_names)}", logger)
    log(f"Test Perturbations: {sorted(data_splitter.test_pert_names)}", logger)
    log(f"Train data size: {len(train_loader.dataset)}", logger)
    log(f"Val data size: {len(val_loader.dataset)}", logger)
    log(f"Test data size: {len(test_loader.dataset)}", logger)

    if args.mode == "train":
        train(model, train_loader, val_loader, n_epochs=args.n_epochs, lr=args.lr, step_size=args.scheduler_step_size,
            device=device, amp=not args.no_amp, early_stopping_patience=args.early_stopping_patience, 
            save_dir=save_dir, logger=logger)

    # Load the best saved model and evaluate on the test set.
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", map_location=device))
    evaluate(model, test_loader, device=device, amp=not args.no_amp, logger=logger)

    log("Done.", logger)

