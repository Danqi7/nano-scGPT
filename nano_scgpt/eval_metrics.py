from typing import Dict, List, Tuple
import numpy as np
import scanpy as sc

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist

def compute_perturbation_metrics(
    preds: np.ndarray, 
    gts: np.ndarray, 
    pert_names:np.ndarray, 
    ctrl_adata: sc.AnnData, 
    subgroups: Dict[str, List[int]] = None,
    control_condition: str = "ctrl") -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
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
        subgroups (`Dict[str, List[int]]`, optional):
            A dictionary where keys are subgroup names and values are lists of perturbation names.
        control_condition (`str`, optional):
            The name of the control condition. Default is "ctrl".
    Returns:
        metrics_across_genes: dict of lists, where each list contains the metric values for each gene, averaged across perturbations.
        metrics_by_perts: dict of dicts, where each key is a perturbation name and each value is a dict of metrics for that perturbation.
        subgroup_metrics: dict of dicts, where each key is a subgroup name and each value is a dict of metrics for that subgroup.
    """
    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
        "pearson_perturbation_distance": 0.0,
        "spearman_perturbation_distance": 0.0,
        "direction_accuracy": 0.0,
        "direction_accuracy_de": 0.0
    } # across gene metrics, averaged across perturbations.
    metrics_by_perts = {} # [pertubation_name] -> dict of metrics for that perturbation.
    subgroup_metrics = {} # [subgroup_name] -> dict of metrics for that subgroup.

    unique_pert_names = np.unique(pert_names)
    assert not control_condition in unique_pert_names, "Control condition should not be included in the perturbation names."
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
    for x, y, name in zip(preds_by_pert[valid], gts_by_pert[valid], unique_pert_names[valid]):
        pc = float(pearsonr(x, y)[0])
        metrics_across_genes["pearson"].append(pc) # [n_perts]
        if name not in metrics_by_perts:
            metrics_by_perts[name] = {}
        metrics_by_perts[name]["pearson"] = pc
    
    # 2. Pearson correlation across genes for delta expression, averaged across perturbations.
    for x, y, name in zip(preds_delta_by_pert[valid], gts_delta_by_pert[valid], unique_pert_names[valid]):
        pc_delta = float(pearsonr(x, y)[0])
        metrics_across_genes["pearson_delta"].append(pc_delta) # [n_perts]
        if name not in metrics_by_perts:
            metrics_by_perts[name] = {}
        metrics_by_perts[name]["pearson_delta"] = pc_delta
    
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
    for x, y, name in zip(preds_by_pert_de[valid_de], gts_by_pert_de[valid_de], unique_pert_names[valid_de]):
        pc_de = float(pearsonr(x, y)[0])
        metrics_across_genes["pearson_de"].append(pc_de) # [n_perts]
        if name not in metrics_by_perts:
            metrics_by_perts[name] = {}
        metrics_by_perts[name]["pearson_de"] = pc_de
    
    # 4. Pearson correlation across DE genes for delta expression.
    for x, y, name in zip(preds_delta_by_pert_de[valid_de], gts_delta_by_pert_de[valid_de], unique_pert_names[valid_de]):
        pc_de_delta = float(pearsonr(x, y)[0])
        metrics_across_genes["pearson_de_delta"].append(pc_de_delta) # [n_perts]
        if name not in metrics_by_perts:
            metrics_by_perts[name] = {}
        metrics_by_perts[name]["pearson_de_delta"] = pc_de_delta

    # 5. Compute correlation on ground truth perturbation distances vs predicted perturbation distances.
    # Compute pairwise distances between perturbations based on their mean expression profiles.
    gt_distances = pdist(gts_by_pert, metric='euclidean') # [n_perts * (n_perts - 1) / 2]
    pred_distances = pdist(preds_by_pert, metric='euclidean') # [n_perts * (n_perts - 1) / 2]
    # Flatten the distance matrices and compute Pearson correlation between them.
    metrics_across_genes["pearson_perturbation_distance"] = float(pearsonr(gt_distances, pred_distances)[0])
    metrics_across_genes["spearman_perturbation_distance"] = float(spearmanr(gt_distances, pred_distances)[0])

    # 6. Direction accuracy: proportion of genes where the predicted change in expression has the same sign as the ground truth change in expression.
    direction_accuracy = np.mean(np.sign(preds_delta_by_pert) == np.sign(gts_delta_by_pert))
    metrics_across_genes["direction_accuracy"] = float(direction_accuracy)
    direction_accuracy_de = np.mean(np.sign(preds_delta_by_pert_de) == np.sign(gts_delta_by_pert_de))
    metrics_across_genes["direction_accuracy_de"] = float(direction_accuracy_de)

    # Subgroup analysis if subgroups are provided.
    if subgroups is not None:
        for subgroup_name, subgroup_perts in subgroups.items():
            sub_pearson = []
            sub_pearson_de = []
            sub_pearson_delta = []
            sub_pearson_de_delta = []
            sub_direction_accuracy = []
            sub_direction_accuracy_de = []
            for pert in subgroup_perts:
                if pert in metrics_by_perts:
                    if "pearson" in metrics_by_perts[pert]:
                        sub_pearson.append(metrics_by_perts[pert]["pearson"])
                    if "pearson_de" in metrics_by_perts[pert]:
                        sub_pearson_de.append(metrics_by_perts[pert]["pearson_de"])
                    if "pearson_delta" in metrics_by_perts[pert]:
                        sub_pearson_delta.append(metrics_by_perts[pert]["pearson_delta"])
                    if "pearson_de_delta" in metrics_by_perts[pert]:
                        sub_pearson_de_delta.append(metrics_by_perts[pert]["pearson_de_delta"])
                    if "direction_accuracy" in metrics_by_perts[pert]:
                        sub_direction_accuracy.append(metrics_by_perts[pert]["direction_accuracy"])
                    if "direction_accuracy_de" in metrics_by_perts[pert]:
                        sub_direction_accuracy_de.append(metrics_by_perts[pert]["direction_accuracy_de"])
            subgroup_metrics[subgroup_name] = {
                "pearson": float(np.mean(sub_pearson)) if sub_pearson else np.nan,
                "pearson_de": float(np.mean(sub_pearson_de)) if sub_pearson_de else np.nan,
                "pearson_delta": float(np.mean(sub_pearson_delta)) if sub_pearson_delta else np.nan,
                "pearson_de_delta": float(np.mean(sub_pearson_de_delta)) if sub_pearson_de_delta else np.nan,
                "direction_accuracy": float(np.mean(sub_direction_accuracy)) if sub_direction_accuracy else np.nan,
                "direction_accuracy_de": float(np.mean(sub_direction_accuracy_de)) if sub_direction_accuracy_de else np.nan,
            }

    metrics_across_genes = {k: float(np.mean(v)) for k, v in metrics_across_genes.items()}

    return metrics_across_genes, metrics_by_perts, subgroup_metrics