from dataclasses import dataclass, field
from typing import Dict, List

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download

def _check_log1ped(data):
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True

class scGPTDataset(Dataset):

    def __init__(self, adata, tokenizer):
        self.adata = adata
        self.tokenizer = tokenizer
        self.gene_names = adata.var['gene_symbol'].tolist()

        self.vocab_genes_idx = [idx for idx, g in enumerate(self.gene_names) if g in self.tokenizer.vocab]
        self.aligned_gene_ids = np.array([self.tokenizer.vocab[self.gene_names[idx]] for idx in self.vocab_genes_idx]) # shape [G_vocab]
        if len(self.aligned_gene_ids) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")

        # TODO: may not be ideal if adata is too large.
        self.X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        
        print(f"Original genes: {len(self.gene_names)}| Genes in vocab: {len(self.aligned_gene_ids)}")
    
    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, idx):
        exprs = self.X[idx]
        exprs = exprs[self.vocab_genes_idx] # shape [G_vocab]
        
        return {
            "exprs": exprs,
            "gene_ids": self.aligned_gene_ids,
        }

    
    def collate_fn(self, batch):
        exprs = np.stack([item["exprs"] for item in batch]) # shape [B, G]
        gene_ids = np.stack([item["gene_ids"] for item in batch]) # shape [G]

        max_expressed = np.max(np.sum(exprs > 0, axis=1)) + 1 # +1 for the [CLS] token
        T = min(max_expressed, self.tokenizer.max_length)

        encoded_cells = [self.tokenizer._encode(exprs[i].squeeze(), gene_ids[i].squeeze(), T) for i in range(len(batch))]

        # Pad and stack the cells into tensors of shape [B, T]
        out_gene_ids = torch.stack([cell["gene_ids"] for cell in encoded_cells]) # shape [B, T]
        out_exprs = torch.stack([cell["exprs"] for cell in encoded_cells]) # shape [B, T]
        out_padding_mask = torch.stack([cell["padding_mask"] for cell in encoded_cells]) # shape [B, T]

        return {
            "gene_ids": out_gene_ids, # shape [B, T]
            "exprs": out_exprs, # shape [B, T]
            "padding_mask": out_padding_mask, # shape [B, T]
        }



@dataclass
class scGPTTokenizerConfig:
    model_type: str = "scGPT_human"

    special_tokens: List[str] = field(default_factory=lambda: ["<cls>", "<pad>", "<eoc>"])
    max_length: int = 1200

    pad_value: float = -2.0
    cls_value: float = -2.0

    sampling: bool = True # Whether to do sampling or simple truncation when len > max_length.

    keep_first_n_tokens: int = 1 # do not bin the first n tokens, here specific refers to [CLS] token appended at the front.
    do_binning: bool = True
    num_bins: int = 51
    binning_deterministic: bool = False # whether deterministic or stochastic binning (i.e., randomize the bin assignment for values that fall on the edge of bins).


class scGPTTokenizer:

    def __init__(self, vocab: Dict[str, int], config):
        self.vocab = vocab
        self.special_tokens = config.special_tokens
        self.max_length = config.max_length
        self.pad_value = config.pad_value
        self.cls_value = config.cls_value
        self.sampling = config.sampling
        self.keep_first_n_tokens = config.keep_first_n_tokens
        self.do_binning = config.do_binning
        self.num_bins = config.num_bins
        self.binning_deterministic = config.binning_deterministic
        

    def encode(self, 
                 exprs: np.ndarray, # shape [B, G]
                 gene_names: List[str], # [G]
                 **kwargs) -> Dict[str, torch.Tensor]:
        # Align Vocab.
        vocab_genes_idx = [idx for idx, g in enumerate(gene_names) if g in self.vocab]

        aligned_exprs = exprs[:, vocab_genes_idx] # shape [B, G_vocab]
        aligned_gene_ids = np.array([self.vocab[gene_names[idx]] for idx in vocab_genes_idx]) # shape [G_vocab]
        if len(aligned_gene_ids) == 0:
            raise ValueError("None of the input genes are in the vocabulary.")
        print(f"Original genes: {len(gene_names)}| Genes in vocab: {len(aligned_gene_ids)}")

        # Determine the max sequence length T for this batch (after filtering zero-expression genes)
        max_expressed = np.max(np.sum(aligned_exprs > 0, axis=1)) + 1 # +1 for the [CLS] token
        T = min(max_expressed, self.max_length)
        print(f"Max expressed genes in this batch (after filtering zero-expression genes): {max_expressed-1}. Using T={T} for encoding.")

        encoded_cells = [self._encode(aligned_expr, aligned_gene_ids, T) for aligned_expr in aligned_exprs]

        # Pad and stack the cells into tensors of shape [B, T]
        out_gene_ids = torch.stack([cell["gene_ids"] for cell in encoded_cells])
        out_exprs = torch.stack([cell["exprs"] for cell in encoded_cells])
        out_padding_mask = torch.stack([cell["padding_mask"] for cell in encoded_cells])

        return {
            "gene_ids": out_gene_ids,
            "exprs": out_exprs,
            "padding_mask": out_padding_mask,
        }
    
    def _encode(self, exprs: np.ndarray, gene_ids: np.ndarray, T: int) -> Dict[str, torch.Tensor]:
        """
        Encode a single cell into gene_ids, exprs, and padding_mask.

        Args:
            exprs (`np.ndarray` of shape `[G_vocab]`): 
                the expression values of the genes in the vocab for this cell.
            gene_ids (`np.ndarray` of shape `[G_vocab]`): 
                the gene IDs of the genes in the vocab for this cell.
            T (`int`):
                the max sequence length for the input batch. The output gene_ids and exprs will be padded/trimmed to this length.
        Returns:
            A dict containing:
            - `gene_ids` (`torch.Tensor` of shape `[T]`): the gene IDs of the tokens for this cell.
            - `exprs` (`torch.Tensor` of shape `[T]`): the expression values of the tokens for this cell.
            - `padding_mask` (`torch.Tensor` of shape `[T]`): a boolean mask indicating which tokens are padding (True for padding, False for real tokens).
        """
        # Non-zero genes & expressions
        non_zero_mask = exprs > 0
        gene_ids = gene_ids[non_zero_mask]
        exprs = exprs[non_zero_mask]

        # Append [CLS] token at the front
        gene_ids = np.concatenate(([self.vocab["<cls>"]], gene_ids))
        exprs = np.concatenate(([self.cls_value], exprs))

        if self.do_binning:
            bins = np.quantile(exprs[self.keep_first_n_tokens:], q=np.linspace(0, 1, self.num_bins-1))
            binned_exprs = self._digitize(exprs[self.keep_first_n_tokens:], bins)
            exprs = np.concatenate((exprs[:self.keep_first_n_tokens], binned_exprs))
        
        out_gene_ids = np.full(T, self.vocab["<pad>"], dtype=np.int64)
        out_exprs = np.full(T, self.pad_value, dtype=np.float32)
        out_padding_mask = np.zeros(T, dtype=bool)

        if len(gene_ids) <= T:
            out_gene_ids[:len(gene_ids)] = gene_ids
            out_exprs[:len(exprs)] = exprs
            out_padding_mask[len(gene_ids):] = True
        else:
            if self.sampling:
                indices = torch.randperm(len(gene_ids) - self.keep_first_n_tokens)[: T - self.keep_first_n_tokens]
                indices = torch.cat([torch.arange(self.keep_first_n_tokens), indices + self.keep_first_n_tokens], dim=0)
                out_gene_ids = gene_ids[indices]
                out_exprs = exprs[indices]
            else:
                out_gene_ids = gene_ids[:T]
                out_exprs = exprs[:T]

        return {
            "gene_ids": torch.from_numpy(out_gene_ids).to(dtype=torch.long),
            "exprs": torch.from_numpy(out_exprs).to(dtype=torch.float32),
            "padding_mask": torch.from_numpy(out_padding_mask).to(dtype=torch.bool),
        }


    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        if self.binning_deterministic:
            return np.digitize(x, bins)   # left edge, no randomization
        left  = np.digitize(x, bins)
        right = np.digitize(x, bins, right=True)
        return np.ceil(np.random.rand(len(x)) * (right - left) + left).astype(np.int64)


    @classmethod
    def from_pretrained(cls, model_type="scGPT_human"):
        if model_type == "scGPT_human":
            config = scGPTTokenizerConfig()
            vocab_file = hf_hub_download(repo_id="wanglab/scGPT-human", filename="vocab.json")
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Supported types: ['scGPT_human']."
            )

        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        for s in config.special_tokens:
            if s not in vocab:
                vocab[s] = len(vocab)
        
        return cls(vocab, config)
    
if __name__ == "__main__":
    from model import scGPTModel
    from scGPT_tokenizer import scGPTTokenizer

    model = scGPTModel.from_pretrained("scGPT_human")
    model.eval()
    model.to('cpu')

    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    genes = ['DUX4L30', 'CTB-52I2.4', 'USP17L16P', 'RPL7P23']
    exprs = np.array([[1.0, 0.0, 23.0, 6.0], [0.0, 0.0, 3.0, 7.0]])
    encoded = tokenizer.encode(exprs, genes)
    embeddings = model.encode(encoded["gene_ids"], encoded["exprs"], encoded["padding_mask"])
    
    print(f'Embeddings shape: {embeddings.shape}')