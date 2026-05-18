import torch
import torch.nn.functional as F
import torchvision
import lightning as L

from main import instantiate_from_config
from contextlib import contextmanager

from taming.modules.diffusionmodules.improved_model import Encoder, Decoder
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.util import requires_grad
from collections import OrderedDict
from taming.modules.ema import LitEma
from taming.modules.vqvae.simvq import LossBreakdown

import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 accumulate_steps = 1,
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                 log_image_every_n_steps = 500,  # 每隔多少步记录一次图片
                 no_vq = False,  # If True, skip quantization and return raw embedding
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(quantconfig)
        # self.quant_conv = torch.nn.Conv2d(ddconfig.z_channels, quantconfig.e_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(quantconfig.e_dim, ddconfig.z_channels, 1)
        self.use_ema = use_ema
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema and stage is None: #no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.accumulate_steps = accumulate_steps
        self.log_image_every_n_steps = log_image_every_n_steps
        self.no_vq = no_vq


        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        # h = self.quant_conv(h)
        
        # If no_vq is True, skip quantization and return raw embedding
        if self.no_vq:
            # Return raw embedding, set other values to zero
            device = h.device
            # Create zero tensors with appropriate shapes
            # info is typically indices, create a zero tensor with same spatial dimensions
            b, c, h_size, w_size = h.shape
            info = torch.zeros((b, h_size, w_size), dtype=torch.long, device=device)
            disentangle_loss = torch.tensor(0.0, device=device)
            loss_breakdown = LossBreakdown(
                per_sample_entropy=torch.tensor(0.0, device=device),
                codebook_entropy=torch.tensor(0.0, device=device),
                commitment=torch.tensor(0.0, device=device),
                avg_probs=torch.tensor(0.0, device=device)
            )
            return h, disentangle_loss, info, loss_breakdown
        
        (quant, disentangle_loss, info), loss_breakdown = self.quantize(h)
        # quant = self.post_quant_conv(quant)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, disentangle_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec
    
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


    def forward(self, input):
        quant, disentangle_loss, indices, loss_break = self.encode(input)
        dec = self.decode(quant)
        # indices is now a combined index: combined_index = index_0 + index_1 * n_e + index_2 * n_e^2 + ...
        # Flatten indices and count unique combined indices
        indices_flat = indices.reshape(-1)
        for ind in indices_flat.unique():
            ind_int = ind.item() if isinstance(ind, torch.Tensor) else int(ind)
            if 0 <= ind_int < len(self.codebook_count):
                self.codebook_count[ind_int] = 1
        return dec, disentangle_loss, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, disentangle_loss, loss_break = self(x)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        
        # optimize generator (disentangle_loss from SimVQ groups orthogonality)
        aeloss, log_dict_ae = self.loss(disentangle_loss, loss_break, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        aeloss = aeloss / self.accumulate_steps
        self.manual_backward(aeloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_gen.step()
            opt_gen.zero_grad()
            if self.scheduler_type != "None":
                scheduler_gen.step()
        
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
            
        # optimize discriminator
        discloss, log_dict_disc = self.loss(disentangle_loss, loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        discloss = discloss / self.accumulate_steps
        self.manual_backward(discloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            if self.scheduler_type != "None":
                scheduler_disc.step()
            
        #if torch.distributed.get_rank() == 0:
        #    print(log_dict_ae, log_dict_disc)

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, outputs, batch, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
            
            
    def on_train_epoch_start(self):
        # Combined index range: 0 to n_e^num_groups - 1
        self.codebook_count = [0] * self.quantize.codebook_size
        
        # Analyze codebook dimensional collapse
        with torch.no_grad():
            # Get codebook: (n_e, e_dim)
            # Check if quantize has embedding_proj (SimVQ) or not (VQ)
            if hasattr(self.quantize, 'embedding_proj'):
                codebook = self.quantize.embedding_proj(self.quantize.embedding.weight)
            else:
                codebook = self.quantize.embedding.weight
            
            # Normalize if l2_norm is enabled
            if hasattr(self.quantize, 'l2_norm') and self.quantize.l2_norm:
                codebook = F.normalize(codebook, p=2, dim=1)
            
            # Analyze codebook
            S_top5, dims_90, dims_99, effective_rank, num_nonzero_singular_values = \
                self.analyze_codebook_dimensional_collapse(codebook)
            
            # Log to wandb if available
            if WANDB_AVAILABLE and self.trainer.is_global_zero:
                log_dict = {
                    "train/codebook_singular_values_top5": S_top5,
                    "train/codebook_dims_90": dims_90,
                    "train/codebook_dims_99": dims_99,
                    "train/codebook_effective_rank": effective_rank,
                    "train/codebook_num_nonzero_singular_values": num_nonzero_singular_values,
                }
                wandb.log(log_dict, step=self.global_step)
        
    def on_validation_epoch_start(self):
        # Combined index range: 0 to n_e^num_groups - 1
        self.codebook_count = [0] * self.quantize.codebook_size

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, disentangle_loss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(disentangle_loss, loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(disentangle_loss, loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
        
        # indices is now a combined index: combined_index = index_0 + index_1 * n_e + index_2 * n_e^2 + ...
        # Flatten indices and count unique combined indices
        indices_flat = indices.reshape(-1)
        for ind in indices_flat.unique():
            ind_int = ind.item() if isinstance(ind, torch.Tensor) else int(ind)
            if 0 <= ind_int < len(self.codebook_count):
                self.codebook_count[ind_int] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))    
    
        # Log images to tensorboard at step 1
        if self.global_step == 1:
            # Normalize images to [0, 1] for tensorboard
            x_normalized = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            x_rec_normalized = (x_rec + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            
            # Clamp to [0, 1] to ensure valid range
            x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
            x_rec_normalized = torch.clamp(x_rec_normalized, 0.0, 1.0)
            
            # Create image grid
            original_grid = torchvision.utils.make_grid(x_normalized, nrow=4)
            recon_grid = torchvision.utils.make_grid(x_rec_normalized, nrow=4)
            

            self.logger.experiment.add_image(f'val{suffix}/images/original', 
                                                original_grid, 
                                                global_step=self.global_step)
            self.logger.experiment.add_image(f'val{suffix}/images/reconstruction',
                                                recon_grid,
                                                global_step=self.global_step)
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _ = self(x)  # returns (dec, disentangle_loss, loss_break)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
                                    