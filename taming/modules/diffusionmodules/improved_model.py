import torch
import torch.nn as nn

def swish(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type="groupnorm", num_groups=32, eps=1e-6, affine=True):
    """
    Creates a normalization layer based on the specified type.
    
    Args:
        in_channels: Number of input channels
        norm_type: Type of normalization layer. Options: "batchnorm", "layernorm", "instancenorm", "groupnorm"
        num_groups: Number of groups for GroupNorm (only used when norm_type="groupnorm")
        eps: Epsilon value for numerical stability
        affine: Whether to use learnable affine parameters
    
    Returns:
        A normalization layer
    """
    norm_type = norm_type.lower()
    
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(in_channels, eps=eps, affine=affine)
    elif norm_type == "layernorm":
        # LayerNorm for 4D input requires shape conversion
        return nn.LayerNorm(in_channels, eps=eps, elementwise_affine=affine)
    elif norm_type == "instancenorm":
        return nn.InstanceNorm2d(in_channels, eps=eps, affine=affine)
    elif norm_type == "groupnorm":
        if in_channels % num_groups != 0:
            # Adjust num_groups to be a divisor of in_channels
            for g in range(num_groups, 0, -1):
                if in_channels % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups, in_channels, eps=eps, affine=affine)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Must be one of: batchnorm, layernorm, instancenorm, groupnorm")


class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 norm_type = "groupnorm",
                 norm_groups = 32
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = Normalize(in_filters, norm_type=norm_type, num_groups=norm_groups)
        self.norm2 = Normalize(out_filters, norm_type=norm_type, num_groups=norm_groups)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)
    

    def forward(self, x, **kwargs):
        residual = x

        # Handle LayerNorm which requires shape conversion for 4D input [B, C, H, W] -> [B, H, W, C]
        if type(self.norm1).__name__ == "LayerNorm":
            x = x.permute(0, 2, 3, 1)
            x = self.norm1(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        
        if type(self.norm2).__name__ == "LayerNorm":
            x = x.permute(0, 2, 3, 1)
            x = self.norm2(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False,
                norm_type="groupnorm", norm_groups=32,
                **ignore_kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        
        # Backward compatibility: if norm_type or norm_groups not provided, use defaults
        self.norm_type = norm_type if norm_type is not None else "groupnorm"
        self.norm_groups = norm_groups if norm_groups is not None else 32
        
        self.conv_in = nn.Conv2d(in_channels,
                                 ch,
                                 kernel_size=(3, 3),
                                 padding=1,
                                 bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] #[1, 1, 2, 2, 4]
            block_out = ch*ch_mult[i_level] #[1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, norm_type=self.norm_type, norm_groups=self.norm_groups))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, norm_type=self.norm_type, norm_groups=self.norm_groups))
        
        ### end
        self.norm_out = Normalize(block_out, norm_type=self.norm_type, num_groups=self.norm_groups)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))
            
    def forward(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level <  self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        ## mid 
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        

        # Handle LayerNorm which requires shape conversion for 4D input [B, C, H, W] -> [B, H, W, C]
        if type(self.norm_out).__name__ == "LayerNorm":
            x = x.permute(0, 2, 3, 1)
            x = self.norm_out(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False,
                norm_type="groupnorm", norm_groups=32,
                **ignore_kwargs) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Backward compatibility: if norm_type or norm_groups not provided, use defaults
        self.norm_type = norm_type if norm_type is not None else "groupnorm"
        self.norm_groups = norm_groups if norm_groups is not None else 32

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, norm_type=self.norm_type, norm_groups=self.norm_groups))
        
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, norm_type=self.norm_type, norm_groups=self.norm_groups))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)
        
        self.norm_out = Normalize(block_in, norm_type=self.norm_type, num_groups=self.norm_groups)

        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)
    
    def forward(self, z):

        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        # Handle LayerNorm which requires shape conversion for 4D input
        if isinstance(self.norm_out, nn.LayerNorm):
            z = z.permute(0, 2, 3, 1)
            z = self.norm_out(z)
            z = z.permute(0, 3, 1, 2)
        else:
            z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x

class Upsampler(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out
        




if __name__ == "__main__":
    x = torch.randn(size = (2, 3, 128, 128))
    encoder = Encoder(ch=128, in_channels=3, num_res_blocks=2, z_channels=18, out_ch=3, resolution=128)
    decoder = Decoder(out_ch=3, z_channels=18, num_res_blocks=2, ch=128, in_channels=3, resolution=128)
    z = encoder(x)
    out = decoder(z)


        


