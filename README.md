# DimVQ: Unveiling And Addressing Dimensional Collapse In Vector Quantization Models Via Codebook Regularization

<p align="center">
  <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/arXiv-TODO-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/jackD/DimVQ"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow" alt="HuggingFace"></a>
  <a href="https://github.com/ksblk2116/dimvq"><img src="https://img.shields.io/badge/GitHub-Code-blue" alt="GitHub"></a>
</p>

Official implementation for the **ICML 2026** paper.

## Overview

We identify **dimensional collapse** in vector quantization models: even with 100% codebook utilization, codebook embeddings degenerate into low-dimensional subspaces. We propose a simple yet effective **codebook regularization** that restores suppressed low-variance components, bridging the spectral gap between discrete and continuous representations.

## Core Method

The key idea is a codebook regularization loss that minimizes off-diagonal correlations of the codebook's Gram matrix, encouraging each dimension to encode diverse information. The implementation is minimal:

```python
import torch
import torch.nn.functional as F

def codebook_regularization_loss(codebook):
    """
    Codebook regularization loss (Eq. in paper).

    Args:
        codebook: (K, D) codebook embedding matrix
    Returns:
        loss: scalar regularization loss
    """
    # Normalize codebook vectors to unit length
    codebook_normalized = F.normalize(codebook, p=2, dim=-1)  # (K, D)

    # Compute Gram matrix: G = C_norm^T @ C_norm
    gram_matrix = torch.mm(codebook_normalized.t(), codebook_normalized)  # (D, D)

    # Off-diagonal mask
    c = gram_matrix.size(0)
    identity = torch.eye(c, device=gram_matrix.device)
    off_diagonal_mask = 1 - identity

    # Loss: mean of squared off-diagonal elements (encourage orthogonality)
    loss = torch.sum((gram_matrix * off_diagonal_mask) ** 2) / (c * (c - 1))

    return loss
```

Full implementation: [`taming/modules/vqvae/simvq.py`](taming/modules/vqvae/simvq.py)

## Model Zoo

### Pre-trained Checkpoints

Available on [HuggingFace](https://huggingface.co/jackD/DimVQ):

| Model | Resolution | K | D | rFID | LPIPS | PSNR | SSIM | Checkpoint |
|-------|-----------|------|-----|------|-------|------|------|-----------|
| SimVQ + Ours | 128x128 | 65,536 | 128 | - | - | - | - | [Download](https://huggingface.co/jackD/DimVQ/tree/main/simvq_K65536) |
| SimVQ + Ours | 128x128 | 262,144 | 128 | 1.55 | 0.10 | 24.99 | 81.2 | [Download](https://huggingface.co/jackD/DimVQ/tree/main/simvq_K262144) |

### TODO

- [ ] Release IBQ checkpoints (K=16384, K=262144, 256x256)
- [ ] Release downstream autoregressive generation models (IBQ-B, IBQ-L, IBQ-XXL)

## Directory Structure

```
DimVQ/
├── main.py                          # Training script (PyTorch Lightning)
├── evaluation.py                    # Evaluation (PSNR, SSIM, LPIPS, FID, Effective Rank)
├── requirements.txt                 # Dependencies
├── configs/
│   ├── imagenet_simvq_K65536_D128_128.yaml   # SimVQ K=65536 D=128 128x128
│   └── imagenet_simvq_K262144_D128_128.yaml  # SimVQ K=262144 D=128 128x128
├── taming/
│   ├── models/
│   │   └── vq.py                    # VQ-VAE model (encoder + quantizer + decoder)
│   ├── modules/
│   │   ├── vqvae/
│   │   │   └── simvq.py            # Core quantizers (SimVQ, VQ, IBQ) + codebook regularization
│   │   ├── diffusionmodules/
│   │   │   └── improved_model.py   # Encoder & Decoder architecture
│   │   ├── discriminator/
│   │   │   └── model.py            # PatchGAN discriminator
│   │   ├── losses/
│   │   │   ├── vqperceptual.py     # Perceptual + GAN loss
│   │   │   └── lpips.py            # LPIPS loss
│   │   ├── scheduler/
│   │   │   └── lr_scheduler.py     # LR schedulers
│   │   ├── ema.py                  # Exponential Moving Average
│   │   └── util.py                 # Utilities
│   └── data/
│       ├── imagenet.py             # ImageNet dataloader
│       ├── base.py                 # Base dataset
│       └── utils.py                # Data utilities
└── metrics/
    ├── inception.py                # InceptionV3 for FID
    └── fid.py                      # FID computation
```

## Installation

```bash
pip install torch torchvision lightning
pip install einops omegaconf lpips scikit-image scipy tqdm
```

## Data Preparation

Organize ImageNet as:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

## Training

```bash
# SimVQ with codebook regularization (K=65536, D=128, 128x128)
python main.py --config configs/imagenet_simvq_K65536_D128_128.yaml

# SimVQ with codebook regularization (K=262144, D=128, 128x128)
python main.py --config configs/imagenet_simvq_K262144_D128_128.yaml
```

Multi-GPU training is handled automatically via PyTorch Lightning DDP. Adjust `trainer.devices` and `trainer.num_nodes` in the config.

## Evaluation

```bash
python evaluation.py \
    --config_file results/<exp_name>/config.yaml \
    --ckpt_path results/<exp_name>/last.ckpt
```

## Citation

```bibtex
@inproceedings{zhang2026dimvq,
  title={Unveiling And Addressing Dimensional Collapse In Vector Quantization Models Via Codebook Regularization},
  author={Zhang, Fang and Zhu, Yongxin and Liu, Yihao and Fu, Bin and Xu, Linli},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgments

This codebase builds upon [VQGAN](https://github.com/CompVis/taming-transformers), [SimVQ](https://github.com/youngsheen/SimVQ), and [IBQ](https://github.com/tencentarc/seed-voken).

## License

This project is released under the MIT License.
