# DimVQ: Unveiling And Addressing Dimensional Collapse In Vector Quantization Models Via Codebook Regularization

Official implementation for the ICML 2026 paper.

## Overview

We identify **dimensional collapse** in vector quantization models: even with 100% codebook utilization, codebook embeddings degenerate into low-dimensional subspaces. We propose a simple yet effective **codebook regularization** that restores suppressed low-variance components, bridging the spectral gap between discrete and continuous representations.

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

## Core Method: Codebook Regularization

The key contribution is in `taming/modules/vqvae/simvq.py`, function `compute_codebook_regularization_loss()`.

Given codebook $C \in \mathbb{R}^{K \times D}$, our regularization minimizes off-diagonal correlations:

$$\mathcal{L}_{cr} = \lambda \sum_{i} \sum_{j \neq i} \text{Cov}(C)_{ij}^2$$

This encourages each dimension to encode diverse information, naturally restoring suppressed low-variance components.

### Key Hyperparameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Codebook size K | `n_e` | 65536 | Number of codebook entries |
| Embedding dim D | `e_dim` | 128 | Codebook embedding dimension |
| Lambda | `disentangle_loss_weight` | 0.001 | Regularization weight (robust in [1e-3, 1e-1]) |
| Loss type | `disentangle_loss_type` | `codebook_orth` | Regularization variant |

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

This codebase builds upon [VQGAN](https://github.com/CompVis/taming-transformers) and [SimVQ](https://github.com/microsoft/SimVQ).

## License

This project is released under the MIT License.
