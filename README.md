## nano-scGPT

The simplest, fastest repository for scGPT inference, (soon) finetuning and trianing, with minimal dependencies. It reimplements the original [scGPT](https://github.com/bowang-lab/scGPT) from scratch. `nano_scgpt/model.py` is pure PyTorch in ~270 lines of code, and `nano_scgpt/scGPT_tokenizer.py` turns raw scRNA data into model input. Small enough to read in one sitting and hack on.

### Why nano-scGPT
Cell modeling is potentially the most exciting and under-indexed AI/ML area. The hope is to make state-of-the-art cell models more accessible to run, understand, and tinker with.

## Install
gi
```bash
git clone https://github.com/Danqi7/nano-scGPT.git
cd nano-scgpt
pip install -e .       # or: uv pip install -e .
```

### Quick Start
```python
# scGPT Embedding example
import numpy as np
from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.model import scGPTModel

model = scGPTModel.from_pretrained("scGPT_human")
model.eval()

tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
genes = ['DUX4L30', 'CTB-52I2.4', 'USP17L16P', 'RPL7P23']
exprs = np.array([[1.0, 0.0, 23.0, 6.0], [0.0, 0.0, 3.0, 7.0]])
encoded = tokenizer.encode(exprs, genes)
embeddings = model.encode(encoded["gene_ids"], encoded["exprs"], encoded["padding_mask"]) # [N, D_embd]

```

### Task: Embed .h5ad scRNA data
```bash
# Example: Tabula Sapiens lung data (downloaded automatically)
python task/embedding.py

# Or on your own local file
python task/embedding.py \
    --input <path to local .h5ad file> \
    --output <path to save embeddings>

# Or from a remote URL
python task/embedding.py \
    --input_url <URL to a remote .h5ad file> \
    --output <path to save embeddings>
```

### todos
- [ ] Finetuning for perturbation response prediction
- [ ] Training from scratch

Let me know what tasks or even models you'd like to see next!

### Acknowledgments
1. This repository reimplements scGPT from scratch. All credit for the original model and method goes to the authors (Cui et al., *Nature Methods*, 2024). See the [original repo](https://github.com/bowang-lab/scGPT) and [paper](https://doi.org/10.1038/s41592-024-02201-0).
2. nano-scGPT is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanogpt) and Chris Hayduk's [minAlphaFold2](https://github.com/ChrisHayduk/minAlphaFold2).
