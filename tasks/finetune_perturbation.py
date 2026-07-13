import torch
import scanpy as sc

from nano_scgpt.model import scGPTForPerturbationResponsePrediction
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.train import PerturbationTrainer, preprocess_adata, log

import os
import json
import argparse
import logging
import requests
from tqdm import tqdm
from pathlib import Path
import zipfile

def _stream_download(url: str, save_path: Path, chunk_size: int = 1024) -> None:
    '''Stream a file from `url` to `save_path` with a progress bar.'''
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=save_path.name
    ) as progress:
        for chunk in response.iter_content(chunk_size):
            f.write(chunk)
            progress.update(len(chunk))

def _download_or_load_data(data_name: str, data_dir: str = './data') -> sc.AnnData:
    '''
    Download the specified perturbation dataset from the given URL, unzip it and return `perturb_processed.h5ad` as an AnnData object.
    
    Adapted from GEAR repo `https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py` .
    '''
    if data_name == 'norman':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
    elif data_name == 'adamson':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
    elif data_name == 'replogle_k562_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
    elif data_name == 'replogle_rpe1_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
    else: 
        raise ValueError(
                f"Unknown data_name '{data_name}'. Expected one of: "
                "'norman', 'adamson', 'replogle_k562_essential', 'replogle_rpe1_essential'"
            )

    data_root = Path(data_dir)
    dataset_dir = data_root / data_name
    zip_path = data_root / f'{data_name}.zip'
    h5ad_path = dataset_dir / 'perturb_processed.h5ad'

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if h5ad_path.exists():
        print(f"Found local copy of '{data_name}' at {h5ad_path}")
    else:
        if not zip_path.exists():
            print(f"Downloading '{data_name}' from {url} ...")
            _stream_download(url, zip_path)
        else:
            print(f"Found existing zip for '{data_name}', skipping download.")

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=data_root)

        if not h5ad_path.exists():
            found = list(dataset_dir.rglob('perturb_processed.h5ad'))
            if not found:
                raise FileNotFoundError(
                    f"'perturb_processed.h5ad' not found after extracting {zip_path} "
                    f"into {dataset_dir}."
                )
            h5ad_path = found[0]

        print("Data downloaded and extracted.")

    adata = sc.read_h5ad(h5ad_path)
    return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adamson", type=str, choices=["adamson", "norman", "replogle_k562_essential", "replogle_rpe1_essential"], help="Dataset to use for training and evaluation.")
    parser.add_argument("--data_dir", default="./data", type=str, help="Directory to store the downloaded data.")
    parser.add_argument("--data_file", default=None, help="Path to the custom data file.")
    parser.add_argument("--gene_symbol_col", default="gene_name", type=str, help="Column name in adata.var that contains gene symbols.")
    parser.add_argument("--condition_col", default="condition", type=str, help="Column name in adata.obs that contains condition names.")
    parser.add_argument("--condition_delimiter", default="+", type=str, help="Delimiter used in condition names to separate multiple genes.")
    parser.add_argument("--control_condition", default="ctrl", type=str, help="Name of the control condition in the dataset.")
    parser.add_argument("--covariate", default="cell_type", type=str, help="Covariate column in adata.obs for DGE analysis.")
    parser.add_argument("--groupby", default="condition_name", type=str, help="Column in adata.obs to group by for DGE analysis.")
    parser.add_argument("--pre_normalized", action='store_true', help="Whether the input data is already normalized. If not, normalization will be applied.")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"], help="Whether to train the model or evaluate the best saved model on the test set.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--scheduler_step_size", type=int, default=1, help="Step size for the learning rate scheduler.")
    parser.add_argument("--no_amp", action='store_true', help="Whether to disable automatic mixed precision (AMP) for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")    
    parser.add_argument("--keep_genes_per_cell", action='store_true', help="Whether to keep genes per cell in the input data. OG scGPT default to False.")
    parser.add_argument("--load_splits", action='store_true', help="Whether to load pre-saved train/val/test splits from a file. If not, new splits will be created and saved.")
    parser.add_argument("--log", action='store_true', help="Whether to log the training and evaluation process to a file.")

    parser.add_argument("--perturbation", type=str, default=None, help="Perturbation to predict. Required if mode is 'predict'.")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples to generate for prediction. Only used if mode is 'predict'.")

    args = parser.parse_args()

    os.makedirs(f"./results/{args.data}", exist_ok=True)
    save_dir = f"./results/{args.data}/keep_genes_per_cell_{args.keep_genes_per_cell}_seed_{args.seed}"
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
    
    if not args.data_file:
        # these are already preprocessed by GEAR.
        print(f"Loading dataset '{args.data}' from {args.data_dir} ...")
        adata = _download_or_load_data(args.data, data_dir=args.data_dir)
        adata.var['gene_symbol'] = adata.var[args.gene_symbol_col]
    else:
        # custom data file provided by user. Preprocess it to match the expected format.
        print(f"Loading custom dataset from {args.data_file} ...")
        adata = sc.read_h5ad(args.data_file)
        adata = preprocess_adata(adata,
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
        print("Starting training...")
        trainer.train()
    
    # Evaluate best model on test set.
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=args.device))
        print("Starting evaluation ...")
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
    
    # Predict perturbation response using the best model.
    elif args.mode == 'predict':
        model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=args.device))
        assert args.perturbation is not None, "Please specify a perturbation for prediction."
        print(f"Starting prediction for perturbation {args.perturbation}...")
        ctrl_adata = trainer.test_dataset.adata[trainer.test_dataset.adata.obs[args.condition_col] == args.control_condition]
        pred = model.predict(perturbation=args.perturbation, 
                                ctrl_adata=ctrl_adata, 
                                tokenizer=trainer.tokenizer,
                                n_samples=args.n_samples,
                                pert_delimiter=args.condition_delimiter,
                                control_condition=args.control_condition)
        print(f"Prediction for perturbation {args.perturbation} completed. Shape of prediction: {pred.shape}")
        if args.save_dir:
            torch.save(pred, f"{args.save_dir}/pred_{args.perturbation}.pt")
            print(f"Prediction saved to {args.save_dir}/pred_{args.perturbation}.pt")
    else:
        raise ValueError(f"Unknown mode '{args.mode}'. Expected one of: 'train', 'eval', 'predict'.")

    print("Done.")
    
