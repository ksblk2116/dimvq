"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""

import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import importlib
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse
import torchvision.utils as vutils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--keep_images", default=10, type=int, help="Number of images to keep and merge into one image")

    return parser.parse_args()

def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.validation.params.config.size = args.image_size
    config_data.data.init_args.batch_size = args.batch_size

    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE) #please specify your own path here
    codebook_size = model.quantize.codebook_size
    #usage
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0

    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(DEVICE)
    inception_model.eval()

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    num_iter = 0
    
    recons_save_dir = Path(args.config_file).parent /  "recons"
    source_save_dir = Path(args.config_file).parent /  "source"
    
    os.makedirs(recons_save_dir, exist_ok=True)
    os.makedirs(source_save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)

            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(images)
                    reconstructed_images = model.decode(quant)
            else:
               quant, diff, indices, _ = model.encode(images)
               reconstructed_images = model.decode(quant)

            reconstructed_images = reconstructed_images.clamp(-1, 1)
            
            ### usage
            for index in indices.flatten():
                usage[index.item()] += 1
            #print(sum([1 for key, value in usage.items() if value > 0]) / codebook_size)
            # calculate lpips
            lpips_alex += loss_fn_alex(images, reconstructed_images).sum()
            lpips_vgg += loss_fn_vgg(images, reconstructed_images).sum()

            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            # calculate fid
            pred_x = inception_model(images)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1
            
            for b in range(0, reconstructed_images.shape[0]):
                vutils.save_image(
                    images[b],
                    os.path.join(source_save_dir, "%s.png"%(num_images + b)),
                    normalize=True,
                    nrow=1,
                )
                vutils.save_image(
                    reconstructed_images[b],
                    os.path.join(recons_save_dir, "%s.png"%(num_images + b)),
                    normalize=True,
                    nrow=1,
                )
            
            num_images += images.shape[0]
    
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)


    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_images
    lpips_vgg_value = lpips_vgg / num_images
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size
    
    # Calculate quant_codebook channel means and projection ranks
    with torch.no_grad():
        # Get codebook from quantize module
        if hasattr(model.quantize, 'get_codebook'):
            codebooks = model.quantize.get_codebook()  # Returns list of codebooks for each group
            # Each codebook is (n_e, group_dim)
            num_groups = len(codebooks)
            
            # Compute mean for each channel position in each group's codebook
            # For each group, compute mean across all n_e codebook vectors for each channel
            channel_means = []
            for g in range(num_groups):
                # For group g, compute mean across all n_e codebook vectors for each channel
                codebook_g_mean = codebooks[g].mean(dim=0)  # (group_dim,)
                channel_means.append(codebook_g_mean)
            
            # Concatenate all group means to get full e_dim channel means
            full_channel_means = torch.cat(channel_means, dim=0)  # (e_dim,)
            channel_means_np = full_channel_means.cpu().numpy()
            
            # Get projection ranks
            if hasattr(model.quantize, 'get_proj_ranks'):
                proj_ranks = model.quantize.get_proj_ranks()  # Returns tensor with ranks
                proj_ranks_np = proj_ranks.cpu().numpy()
            else:
                proj_ranks_np = None
        else:
            channel_means_np = None
            proj_ranks_np = None
    
    def print_and_save(message, file):
        print(message)  
        file.write(message + '\n') 
    
    with open(Path(args.ckpt_path).parent / "result.txt", 'w') as f:
        print_and_save(f"FID: {fid_value}", f)
        print_and_save(f"LPIPS_ALEX: {lpips_alex_value.item()}", f)
        print_and_save(f"LPIPS_VGG: {lpips_vgg_value.item()}", f)
        print_and_save(f"SSIM: {ssim_value}", f)
        print_and_save(f"PSNR: {psnr_value}", f)
        print_and_save(f"utilization: {utilization}", f)
        
        # Add channel means
        if channel_means_np is not None:
            print_and_save(f"\nCodebook Channel Means (per channel average across all codebook vectors):", f)
            print_and_save(f"Mean of channel means: {channel_means_np.mean():.6f}", f)
            print_and_save(f"Std of channel means: {channel_means_np.std():.6f}", f)
            print_and_save(f"Min channel mean: {channel_means_np.min():.6f}", f)
            print_and_save(f"Max channel mean: {channel_means_np.max():.6f}", f)
            # Optionally save all channel means
            channel_means_str = ", ".join([f"{x:.6f}" for x in channel_means_np])
            print_and_save(f"All channel means: [{channel_means_str}]", f)
        
        # Add projection ranks
        if proj_ranks_np is not None:
            print_and_save(f"\nProjection Matrix Ranks:", f)
            if len(proj_ranks_np) > 1:
                # Last one is stacked rank, others are individual group ranks
                for i in range(len(proj_ranks_np) - 1):
                    print_and_save(f"Group {i} projection rank: {proj_ranks_np[i]:.1f}", f)
                print_and_save(f"Stacked projection rank: {proj_ranks_np[-1]:.1f}", f)
            else:
                print_and_save(f"Projection rank: {proj_ranks_np[0]:.1f}", f)
    
    # Merge and keep only specified number of images
    keep_images = min(args.keep_images, num_images)
    print(f"\nMerging {keep_images} images and cleaning up...")
    
    # Load images to merge
    images_to_merge = []
    for i in range(keep_images):
        source_path = source_save_dir / f"{i}.png"
        recon_path = recons_save_dir / f"{i}.png"
        
        if source_path.exists() and recon_path.exists():
            # Load images
            source_img = Image.open(source_path).convert('RGB')
            recon_img = Image.open(recon_path).convert('RGB')
            
            # Convert to tensor and normalize to [-1, 1]
            source_tensor = torch.from_numpy(np.array(source_img)).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
            recon_tensor = torch.from_numpy(np.array(recon_img)).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
            
            # Stack source and reconstructed side by side
            pair_tensor = torch.cat([source_tensor, recon_tensor], dim=2)  # Concatenate horizontally
            images_to_merge.append(pair_tensor)
    
    if images_to_merge:
        # Merge all images into a grid
        # Each element in images_to_merge is already a pair (source | recon) side by side
        merged_tensor = torch.stack(images_to_merge, dim=0)
        # Create grid: arrange pairs vertically (nrow=1 means one pair per row)
        # Each row shows: [original | reconstructed]
        grid = vutils.make_grid(merged_tensor, nrow=1, normalize=True, padding=2)
        
        # Save merged image
        merged_path = Path(args.config_file).parent / "merged_samples.png"
        vutils.save_image(grid, merged_path, normalize=True)
        print(f"Saved merged image with {keep_images} sample pairs to {merged_path}")
    
    # Delete all individual image files
    print("Deleting individual image files...")
    for img_file in source_save_dir.glob("*.png"):
        img_file.unlink()
    for img_file in recons_save_dir.glob("*.png"):
        img_file.unlink()
    print("Cleanup completed.")
  
if __name__ == "__main__":
    args = get_args()
    main(args)