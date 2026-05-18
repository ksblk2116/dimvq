"""
SimVQ: Simplified Vector Quantization with Learnable Projection

This module implements the core SimVQ quantizer with orthogonal regularization
on codebook vectors to prevent dimensional collapse.

Key Features:
1. Learnable projection layer between embedding and codebook
2. Codebook regularization loss preventing dimensional collapse
3. Barlow Twins loss for covariance-based regularization
4. L2 normalization support for embeddings
5. Flexible initialization (Gaussian or Spherical)

Reference: [Your Paper Title]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import namedtuple

LossBreakdown = namedtuple('LossBreakdown', 
                           ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs'])





def analyze_codebook_dimensional_collapse(self, codebook):
    """
    分析码本的向量维度坍缩情况
    
    Args:
        codebook: (n_e, e_dim) normalized codebook tensor
    
    Returns:
        (S_top5, dims_90, dims_99, effective_rank, num_nonzero_singular_values)
    """
    K, D = codebook.shape

    # 预处理：去中心化 (Centering)
    codebook_centered = codebook - codebook.mean(dim=0, keepdim=True)

    # 奇异值分解 (SVD)
    _, S, _ = torch.linalg.svd(codebook_centered, full_matrices=False)
    
    # 计算非零奇异值的个数（使用一个小的阈值，比如 1e-10）
    num_nonzero_singular_values = (S > 0.01).sum().item()
    
    # 转为 numpy 方便计算
    S = S.cpu().numpy()
    
    # 计算指标
    # 解释方差比 (Explained Variance Ratio)
    eigenvalues = S ** 2
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # 累积方差
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 计算需要多少个维度才能解释 90% 和 99% 的信息
    dims_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    dims_99 = np.searchsorted(cumulative_variance, 0.99) + 1
    
    # 有效秩 (Effective Rank)
    p = S / np.sum(S)
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)

    # 返回 S 的前5个值、dims_90、dims_99、有效秩和非零奇异值个数
    S_top5 = S[:5].tolist()
    return S_top5, dims_90, dims_99, effective_rank, num_nonzero_singular_values


# ============================================================================
# Core Loss Functions
# ============================================================================

def compute_codebook_regularization_loss(codebook=None, z_q=None, loss_type="codebook_regularization"):
    """
    Unified function to compute various codebook regularization losses.
    
    This function supports multiple regularization strategies to prevent dimensional
    collapse and improve codebook utilization.
    
    Args:
        codebook: (n_e, e_dim) codebook tensor, where n_e is codebook size
                 and e_dim is embedding dimension. Required for 'codebook_regularization' 
                 and 'barlow_twins_codebook' loss types.
        z_q: (batch*h*w, e_dim) or (batch, h, w, e_dim) quantized vectors.
             Required for 'barlow_twins_zq' loss type.
        loss_type: Type of regularization loss to compute:
            - 'codebook_regularization': Orthogonal regularization on codebook 
                                        (encourages orthogonal codebook vectors)
            - 'barlow_twins_codebook': Barlow Twins loss on codebook (encourages
                                      covariance matrix close to identity)
            - 'barlow_twins_zq': Barlow Twins loss on quantized vectors
    
    Returns:
        loss: scalar loss value
        
    Loss Types Explained:
    
    1. Codebook Regularization Loss (loss_type='codebook_regularization'):
       Encourages codebook vectors to be orthogonal to each other.
       
       Mathematical Formulation:
           G = C_norm^T @ C_norm   (Gram matrix)
           loss = mean((G * (1 - I))^2)  (mean squared off-diagonal elements)
       
       where C_norm is L2-normalized codebook, I is identity matrix.
       
       This prevents dimensional collapse by ensuring codebook vectors span
       different directions in the embedding space.
       
    2. Barlow Twins Loss on Codebook (loss_type='barlow_twins_codebook'):
       Encourages the covariance matrix of codebook to be close to identity.
       This prevents dimensional collapse by ensuring all dimensions have
       similar variance and are uncorrelated.
       
       Mathematical Formulation:
           C_std = (C - mean(C)) / std(C)  (standardize codebook)
           Cov = C_std^T @ C_std / n_e     (covariance matrix)
           loss = sum((Cov - I)^2)         (MSE to identity)
       
       Diagonal terms: encourage variance = 1 (prevent dimension suppression)
       Off-diagonal terms: encourage correlation = 0 (prevent redundancy)
       
    3. Barlow Twins Loss on Quantized Vectors (loss_type='barlow_twins_zq'):
       Same as above but applied to quantized vectors z_q instead of codebook.
       Useful for encouraging diverse usage patterns.
    """
    if loss_type == "codebook_regularization":
        if codebook is None:
            raise ValueError("codebook is required for codebook_regularization loss")
        
        # Normalize codebook vectors to unit length
        codebook_normalized = F.normalize(codebook, p=2, dim=-1)  # (n_e, e_dim)
        
        # Compute Gram matrix: G = C^T @ C
        # This measures similarity between all pairs of codebook vectors
        gram_matrix = torch.mm(codebook_normalized.t(), codebook_normalized)  # (e_dim, e_dim)
        
        # Create identity matrix and off-diagonal mask
        c = gram_matrix.size(0)
        identity = torch.eye(c, device=gram_matrix.device)
        off_diagonal_mask = 1 - identity
        
        # Compute loss: mean of squared off-diagonal elements
        # This encourages all off-diagonal elements to be close to 0 (orthogonal)
        loss = torch.sum((gram_matrix * off_diagonal_mask) ** 2) / (c * (c - 1))
        
        return loss
    
    elif loss_type == "barlow_twins_codebook":
        if codebook is None:
            raise ValueError("codebook is required for barlow_twins_codebook loss")
        
        # Standardize codebook: zero mean, unit variance per dimension
        # Shape: (n_e, e_dim)
        codebook_mean = codebook.mean(dim=0, keepdim=True)  # (1, e_dim)
        codebook_std = codebook.std(dim=0, keepdim=True) + 1e-8  # (1, e_dim)
        codebook_standardized = (codebook - codebook_mean) / codebook_std  # (n_e, e_dim)
        
        # Compute covariance matrix: Cov = C^T @ C / n_e
        n_e = codebook.shape[0]
        covariance_matrix = torch.mm(codebook_standardized.t(), 
                                     codebook_standardized) / n_e  # (e_dim, e_dim)
        
        # Target: identity matrix (diagonal = 1, off-diagonal = 0)
        e_dim = codebook.shape[1]
        identity = torch.eye(e_dim, device=codebook.device)
        
        # Barlow Twins loss: MSE between covariance and identity
        # Diagonal terms: encourage variance = 1 (prevent dimension suppression)
        # Off-diagonal terms: encourage correlation = 0 (prevent redundancy)
        loss = torch.mean((covariance_matrix - identity) ** 2)
        
        return loss
    
    elif loss_type == "barlow_twins_zq":
        if z_q is None:
            raise ValueError("z_q is required for barlow_twins_zq loss")
        
        # Reshape z_q to (N, e_dim) if needed
        if z_q.dim() == 4:
            # (batch, h, w, e_dim) or (batch, e_dim, h, w)
            if z_q.shape[1] > z_q.shape[-1]:  # Likely (batch, e_dim, h, w)
                z_q = z_q.permute(0, 2, 3, 1)  # -> (batch, h, w, e_dim)
            z_q_flat = z_q.reshape(-1, z_q.shape[-1])  # (N, e_dim)
        else:
            z_q_flat = z_q  # Already (N, e_dim)
        
        # Standardize z_q: zero mean, unit variance per dimension
        z_q_mean = z_q_flat.mean(dim=0, keepdim=True)  # (1, e_dim)
        z_q_std = z_q_flat.std(dim=0, keepdim=True) + 1e-8  # (1, e_dim)
        z_q_standardized = (z_q_flat - z_q_mean) / z_q_std  # (N, e_dim)
        
        # Compute covariance matrix
        N = z_q_flat.shape[0]
        covariance_matrix = torch.mm(z_q_standardized.t(), 
                                     z_q_standardized) / N  # (e_dim, e_dim)
        
        # Target: identity matrix
        e_dim = z_q_flat.shape[1]
        identity = torch.eye(e_dim, device=z_q_flat.device)
        
        # Barlow Twins loss
        loss = torch.mean((covariance_matrix - identity) ** 2)
        
        return loss
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Supported: 'codebook_regularization', 'barlow_twins_codebook', 'barlow_twins_zq'")


# ============================================================================
# SimVQ Module
# ============================================================================

class SimVQ(nn.Module):
    """
    Simplified Vector Quantization with Learnable Projection.
    
    This module implements a vector quantizer with a learnable projection layer
    and orthogonal regularization to prevent codebook collapse.
    
    Architecture:
        embedding (frozen) -> projection (learnable) -> quantization
    
    Args:
        n_e (int): Codebook size (number of embeddings)
        e_dim (int): Embedding dimension
        beta (float): Commitment loss weight (default: 0.25)
        embedding_init (str): Initialization method ('gaussian' or 'spherical')
        l2_norm (bool): Whether to L2-normalize embeddings before quantization
        num_groups (int): Number of groups for multi-group quantization
        disentangle_loss_type (str): Type of regularization loss
            - 'codebook_orth': Codebook regularization (orthogonality-based, recommended)
            - 'barlow_twins_codebook': Barlow Twins on codebook (covariance-based)
            - 'barlow_twins_zq': Barlow Twins on quantized vectors
        disentangle_loss_weight (float): Weight for regularization loss
        
    Key Features:
        1. Frozen embedding layer + learnable projection
        2. Orthogonal regularization prevents dimensional collapse
        3. Flexible normalization and initialization
        4. Compatible with standard VQ-VAE training
    """
    
    def __init__(self, n_e, e_dim, beta=0.25, 
                 embedding_init="gaussian",
                 l2_norm=False,
                 num_groups=1,
                 disentangle_loss_type="codebook_orth", 
                 disentangle_loss_weight=0.001,
                 legacy=True,
                 **kwargs):
        super().__init__()
        
        self.n_e = n_e
        self.codebook_size = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding_init = embedding_init
        self.l2_norm = l2_norm
        self.num_groups = num_groups
        self.disentangle_loss_type = disentangle_loss_type
        self.disentangle_loss_weight = disentangle_loss_weight
        
        # Validate loss type
        assert disentangle_loss_type in ["codebook_orth", "barlow_twins_codebook", "barlow_twins_zq"], \
            f"Invalid disentangle_loss_type: {disentangle_loss_type}"
        
        # Frozen embedding layer
        self.embedding = nn.Embedding(n_e, e_dim)
        self._initialize_embedding()
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Learnable projection layer
        self.embedding_proj = nn.Linear(e_dim, e_dim)
        
    def _initialize_embedding(self):
        """Initialize embedding based on specified method."""
        if self.embedding_init == "gaussian":
            nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        elif self.embedding_init == "spherical":
            codebook = torch.randn(self.n_e, self.e_dim)
            codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.embedding.weight.data = codebook
        else:
            raise ValueError(f"Unknown embedding_init: {self.embedding_init}")
    
    def _compute_disentangle_loss(self, quant_codebook, z_q, device):
        """Compute regularization loss based on type."""
        if self.disentangle_loss_weight <= 0:
            return torch.tensor(0.0, device=device)
        
        if self.disentangle_loss_type == "codebook_orth":
            return compute_codebook_regularization_loss(
                codebook=quant_codebook, loss_type="codebook_regularization")
        elif self.disentangle_loss_type == "barlow_twins_codebook":
            return compute_codebook_regularization_loss(
                codebook=quant_codebook, loss_type="barlow_twins_codebook")
        elif self.disentangle_loss_type == "barlow_twins_zq":
            return compute_codebook_regularization_loss(
                z_q=z_q, loss_type="barlow_twins_zq")
        # Other loss types omitted for brevity (see full implementation)
        
        return torch.tensor(0.0, device=device)
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        """
        Forward pass: quantize input z using learned codebook.
        
        Args:
            z: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Tuple of:
                - (z_q, total_loss, indices): Quantized output, loss, and indices
                - LossBreakdown: Breakdown of losses for logging
        """
        # Reshape z -> (batch, height, width, channel)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Get projected codebook
        quant_codebook = self.embedding_proj(self.embedding.weight)  # (n_e, e_dim)
        
        # Optional L2 normalization
        if self.l2_norm:
            quant_codebook = F.normalize(quant_codebook, p=2, dim=1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        
        # Compute distances: d = ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z,e>
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, 
                        rearrange(quant_codebook, 'n d -> d n'))
        
        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        
        # Commitment loss (VQ-VAE loss)
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                         torch.mean((z_q - z.detach()) ** 2)
        else:
            commit_loss = torch.mean((z_q.detach() - z) ** 2) + \
                         self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # Regularization loss (only during training)
        disentangle_loss = torch.tensor(0.0, device=z.device)
        if self.training:
            disentangle_loss = self._compute_disentangle_loss(
                quant_codebook, z_q, z.device)
        
        weighted_disentangle_loss = self.disentangle_loss_weight * disentangle_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Reshape back to original format
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        # Total loss
        total_loss = commit_loss + weighted_disentangle_loss
        
        return (z_q, total_loss, min_encoding_indices), \
               LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), 
                           commit_loss, torch.tensor(0.0))
    
    def get_codebook_entry(self, indices, shape):
        """
        Get quantized vectors for given indices.
        
        Args:
            indices: Codebook indices
            shape: Target shape (batch, height, width, channel)
        
        Returns:
            z_q: Quantized vectors in shape (batch, channel, height, width)
        """
        quant_codebook = self.embedding_proj(self.embedding.weight)
        if self.l2_norm:
            quant_codebook = F.normalize(quant_codebook, p=2, dim=1)
        
        z_q = F.embedding(indices, quant_codebook)
        
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q


# ============================================================================
# Standard VQ (Without Projection) for Comparison
# ============================================================================

class VQ(nn.Module):
    """
    Standard VQ-VAE quantizer without projection layer.
    
    This serves as a baseline comparison to SimVQ. The codebook is directly
    trained without a projection layer.
    
    Args: Same as SimVQ but without projection layer
    """
    
    def __init__(self, n_e, e_dim, beta=0.25,
                 embedding_init="gaussian",
                 l2_norm=False,
                 disentangle_loss_type="codebook_orth",
                 disentangle_loss_weight=0.001,
                 legacy=True,
                 **kwargs):
        super().__init__()
        
        self.n_e = n_e
        self.codebook_size = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding_init = embedding_init
        self.l2_norm = l2_norm
        self.disentangle_loss_type = disentangle_loss_type
        self.disentangle_loss_weight = disentangle_loss_weight
        
        # Trainable embedding (no projection layer)
        self.embedding = nn.Embedding(n_e, e_dim)
        self._initialize_embedding()
        
    def _initialize_embedding(self):
        """Initialize embedding based on specified method."""
        if self.embedding_init == "gaussian":
            nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        elif self.embedding_init == "spherical":
            codebook = torch.randn(self.n_e, self.e_dim)
            codebook = codebook / codebook.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.embedding.weight.data = codebook
        else:
            raise ValueError(f"Unknown embedding_init: {self.embedding_init}")
    
    def _compute_disentangle_loss(self, quant_codebook, z_q, device):
        """Compute regularization loss."""
        if self.disentangle_loss_weight <= 0:
            return torch.tensor(0.0, device=device)
        
        if self.disentangle_loss_type == "codebook_orth":
            return compute_codebook_regularization_loss(
                codebook=quant_codebook, loss_type="codebook_regularization")
        elif self.disentangle_loss_type == "barlow_twins_codebook":
            return compute_codebook_regularization_loss(
                codebook=quant_codebook, loss_type="barlow_twins_codebook")
        elif self.disentangle_loss_type == "barlow_twins_zq":
            return compute_codebook_regularization_loss(
                z_q=z_q, loss_type="barlow_twins_zq")
        
        return torch.tensor(0.0, device=device)
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        """Forward pass: quantize using direct codebook."""
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Use embedding weight directly as codebook (no projection)
        quant_codebook = self.embedding.weight
        
        if self.l2_norm:
            quant_codebook = F.normalize(quant_codebook, p=2, dim=1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        
        # Compute distances and find nearest entries
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened,
                        rearrange(quant_codebook, 'n d -> d n'))
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        
        # Commitment loss
        if not self.legacy:
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                         torch.mean((z_q - z.detach()) ** 2)
        else:
            commit_loss = torch.mean((z_q.detach() - z) ** 2) + \
                         self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # Regularization loss
        disentangle_loss = torch.tensor(0.0, device=z.device)
        if self.training:
            disentangle_loss = self._compute_disentangle_loss(
                quant_codebook, z_q, z.device)
        
        weighted_disentangle_loss = self.disentangle_loss_weight * disentangle_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        total_loss = commit_loss + weighted_disentangle_loss
        
        return (z_q, total_loss, min_encoding_indices), \
               LossBreakdown(torch.tensor(0.0), torch.tensor(0.0),
                           commit_loss, torch.tensor(0.0))
    
    def get_codebook_entry(self, indices, shape):
        """Get quantized vectors for given indices."""
        quant_codebook = self.embedding.weight
        if self.l2_norm:
            quant_codebook = F.normalize(quant_codebook, p=2, dim=1)
        
        z_q = F.embedding(indices, quant_codebook)
        
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q


# ============================================================================
# IBQ: Index Propagation Quantization (Baseline Comparison)
# ============================================================================

def compute_entropy_loss(logits, temperature=0.01, 
                        sample_minimization_weight=1.0, 
                        batch_maximization_weight=1.0):
    """
    Compute entropy-based regularization loss.
    
    Args:
        logits: (N, n_e) logits for each sample
        temperature: Temperature for softmax
        sample_minimization_weight: Weight for per-sample entropy minimization
        batch_maximization_weight: Weight for batch-level entropy maximization
    
    Returns:
        sample_entropy: Average per-sample entropy
        avg_entropy: Batch-level entropy
        entropy_loss: Combined entropy loss
    """
    probs = F.softmax(logits / temperature, dim=1)
    
    # Per-sample entropy (minimize for confident predictions)
    sample_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    avg_sample_entropy = torch.mean(sample_entropy)
    
    # Batch-level entropy (maximize for diverse codebook usage)
    avg_probs = torch.mean(probs, dim=0)
    batch_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
    
    # Combined loss: minimize per-sample entropy, maximize batch entropy
    entropy_loss = sample_minimization_weight * avg_sample_entropy - \
                   batch_maximization_weight * batch_entropy
    
    return avg_sample_entropy, batch_entropy, entropy_loss


class IBQ(nn.Module):
    """
    Index-Based Quantization with Soft Assignment.
    
    This implements a differentiable quantization method using soft assignment
    and straight-through estimator. Serves as a baseline comparison to SimVQ.
    
    Key Differences from SimVQ:
        - Uses soft assignment with straight-through estimator
        - Optional entropy regularization for codebook utilization
        - Direct codebook training (no projection layer)
    
    Args:
        n_e (int): Codebook size
        e_dim (int): Embedding dimension
        beta (float): Commitment loss weight
        use_entropy_loss (bool): Whether to use entropy regularization
        cosine_similarity (bool): Whether to use cosine similarity (not implemented)
        entropy_temperature (float): Temperature for entropy computation
        sample_minimization_weight (float): Weight for per-sample entropy minimization
        batch_maximization_weight (float): Weight for batch entropy maximization
    """
    
    def __init__(self, n_e, e_dim, beta=0.25, 
                 use_entropy_loss=False,
                 cosine_similarity=False,
                 entropy_temperature=0.01,
                 sample_minimization_weight=1.0, 
                 batch_maximization_weight=1.0,
                 **kwargs):
        super().__init__()
        
        self.n_e = n_e
        self.codebook_size = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_entropy_loss = use_entropy_loss
        self.cosine_similarity = cosine_similarity
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight
        
        self.embedding = nn.Embedding(n_e, e_dim)
    
    def forward(self, z, temp=None, return_logits=False):
        """
        Forward pass with soft assignment and straight-through estimator.
        
        Args:
            z: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Tuple of:
                - z_q: Quantized output
                - diff: Loss (or tuple of losses if entropy loss enabled)
                - (None, None, ind): Tuple with indices
        """
        # Compute logits: similarity between z and codebook
        logits = einsum('b d h w, n d -> b n h w', z, self.embedding.weight)
        
        # Soft assignment via softmax
        soft_one_hot = F.softmax(logits, dim=1)
        
        # Hard assignment: argmax
        ind = soft_one_hot.max(dim=1, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(logits, 
                                        memory_format=torch.legacy_contiguous_format
                                        ).scatter_(1, ind, 1.0)
        
        # Straight-through estimator
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot
        
        # Quantize using soft assignment (for gradients)
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embedding.weight)
        
        # Quantize using hard assignment (for loss)
        z_q_hard = einsum('b n h w, n d -> b d h w', hard_one_hot, self.embedding.weight)
         
        # Regularization loss
        disentangle_loss = torch.tensor(0.0, device=z.device)
        if self.training:
            disentangle_loss = self._compute_disentangle_loss(
                quant_codebook, z_q, z.device)
        
        weighted_disentangle_loss = self.disentangle_loss_weight * disentangle_loss
        
        # Quantization loss
        quant_loss = torch.mean((z_q - z) ** 2) + \
                    torch.mean((z_q_hard.detach() - z) ** 2) + \
                    self.beta * torch.mean((z_q_hard - z.detach()) ** 2)
        
        diff = quant_loss
        
        # Optional entropy regularization
        if self.use_entropy_loss:
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, self.n_e)
            sample_entropy, avg_entropy, entropy_loss = compute_entropy_loss(
                logits=logits_flat,
                temperature=self.entropy_temperature,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight
            )
            diff = (quant_loss, sample_entropy, avg_entropy, entropy_loss)
        
        ind = torch.flatten(ind)
        
        return z_q, diff, (None, None, ind)
    
    def get_codebook_entry(self, indices, shape):
        """
        Get quantized vectors for given indices.
        
        Args:
            indices: Codebook indices
            shape: Target shape (batch, height, width, channel)
        
        Returns:
            z_q: Quantized vectors
        """
        z_q = self.embedding(indices)
        
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q
