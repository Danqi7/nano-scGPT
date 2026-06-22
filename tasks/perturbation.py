import torch
from torch.utils.data import DataLoader
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr

from nano_scgpt.model import scGPTForPerturbationResponsePrediction
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.perturbation_data import PerturbationDataSplitter, PerturbationDataset

import warnings
import argparse

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

    # Pearson correlation across genes, averaged across perturbations.
    valid = gts_by_pert.sum(axis=1) != 0
    for x, y in zip(preds_by_pert[valid], gts_by_pert[valid]):
        metrics_across_genes["pearson"].append(pearsonr(x, y)[0]) # [n_perts]
    
    # Pearson correlation across genes for delta expression, averaged across perturbations.
    valid = gts_delta_by_pert.sum(axis=1) != 0
    for x, y in zip(preds_delta_by_pert[valid], gts_delta_by_pert[valid]):
        # import pdb; pdb.set_trace()
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

    # Pearson correlation across DE genes.
    valid = gts_by_pert_de.sum(axis=1) != 0
    for x, y in zip(preds_by_pert_de[valid], gts_by_pert_de[valid]):
        metrics_across_genes["pearson_de"].append(pearsonr(x, y)[0]) # [n_perts]
    
    # Pearson correlation across DE genes for delta expression.
    valid = gts_delta_by_pert_de.sum(axis=1) != 0
    for x, y in zip(preds_delta_by_pert_de[valid], gts_delta_by_pert_de[valid]):
        metrics_across_genes["pearson_de_delta"].append(pearsonr(x, y)[0]) # [n_perts]


    metrics_across_genes = {k: np.mean(v) for k, v in metrics_across_genes.items()}

    return metrics_across_genes



def train(model, train_loader, val_loader, n_epochs=15, lr=1e-4, device='cuda', amp=True, early_stopping_patience=5):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler    = torch.amp.GradScaler(enabled=amp)

    patience = 0
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
            print(f"Epoch {epoch+1}/{n_epochs}, Step {idx+1} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.4f} | Scaler: {scaler.get_scale()} | Norm: {norm:.4f}")

            # break;

        scheduler.step()
    
        # Evaluation.
        model.eval()
        ctrl_adata = val_loader.dataset.adata[val_loader.dataset.adata.obs['condition'] == 'ctrl']
        predictions = []
        gts = []
        perts = []
        best_val_metric = -float('inf')
        with torch.no_grad():
            val_loss = 0.0
            for idx, batch in enumerate(val_loader):
                gene_ids = batch["gene_ids"].to(device)
                gene_values = batch["gene_values"].to(device)
                src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                pert_labels = batch["pert_labels"].to(device)
                target_values = batch["target_values"].to(device)

                with torch.amp.autocast(device_type=device, enabled=amp):
                    pred = model(gene_ids, gene_values, src_key_padding_mask, pert_labels) # (B, n_genes)
                    loss = torch.nn.functional.mse_loss(pred, target_values, reduction='mean')
                    
                    predictions.extend(pred.detach().cpu().numpy())
                    gts.extend(target_values.detach().cpu().numpy())
                    perts.extend(batch["perturbations"])
                
                val_loss += loss.item()

                # if idx > 2:
                #     break;

            val_loss /= len(val_loader)
            predictions = np.stack(predictions)
            gts = np.stack(gts)
            metrics = compute_perturbation_metrics(predictions, gts, np.array(perts), ctrl_adata)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss}, Prediction Shape: {predictions.shape}, GT Shape: {gts.shape}, Perturbations Length: {len(perts)}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            
            # Early stopping based on pearson correlation across genes.
            if metrics["pearson"] > best_val_metric:
                best_val_metric = metrics["pearson"]
                torch.save(model.state_dict(), "best_model.pt")
                print(f"New best model saved with {metric_name}: {metric_value:.4f}")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
    
def evaluate(model, test_loader, device='cuda', amp=True):
    model = model.to(device)
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
        print(f"Test Prediction Shape: {predictions.shape}, GT Shape: {gts.shape}, Perturbations Length: {len(perts)}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Whether to train the model or evaluate the best saved model on the test set.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--amp", action='store_true', help="Whether to use automatic mixed precision (AMP) for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs to wait for improvement before early stopping.")
    args = parser.parse_args()

    batch_size = args.batch_size
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    if args.mode == "train":
        train(model, train_loader, val_loader, n_epochs=args.n_epochs, lr=args.lr, device=device, amp=args.amp, early_stopping_patience=args.early_stopping_patience)

    # Load the best saved model and evaluate on the test set.
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    evaluate(model, test_loader, device=device, amp=args.amp)

    print("Done.")


