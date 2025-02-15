from torch import nn

class PatchEmbedding(nn.Module):
    """
    Makes input's embedding by patching it to the windows of the same size
    and convoluting over each window
    """
    def __init__(self, embed_dim : int = 768, patch_size : int = 7, in_channels : int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        # x : [B, embed_dim, H, W]
        _, _, height, width = x.shape

        if height % self.patch_size != 0:
            padded_height = (height + self.patch_size - 1) // self.patch_size * self.patch_size
            x = nn.functional.pad(x, (0, 0, 0, padded_height-height))
        if width % self.patch_size != 0:
            padded_width = (width + self.patch_size - 1) // self.patch_size * self.patch_size
            x = nn.functional.pad(x, (0, padded_width-width))

        x = self.proj(x) # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2) # [B, embed_dim, N=H/patch_size*W/patch_size]
        x = x.transpose(1,2) # [B, N, embed_dim]
        return x

class SwinBlock(nn.Module):
    """
    Realization of Swin Transformer block
    """
    def __init__(self, embed_dim : int, in_channels : int = 3, patch_size : int = 7, num_heads : int = 8,  mlp_ratio : int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(embed_dim, patch_size, in_channels)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        )

    def forward(self, x):
        # x : [B, embed_dim, H, W]
        batch_size, embed_dim, height, width = x.shape
        patched_height = (height + self.patch_size - 1) // self.patch_size
        patched_width = (width + self.patch_size - 1) // self.patch_size
        x = self.patch_embed(x) # [B, N, embed_dim]
        x_norm1 = self.layer_norm1(x)
        attention_output, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x = x + attention_output
        x_norm2 = self.layer_norm2(x)
        x = x + self.feed_forward(x_norm2)
        x = x.transpose(1, 2) # [B, embed_dim, N]
        x = x.view(batch_size, embed_dim, patched_height, patched_width) # [B, embed_dim, H, W]
        return x

class SwinStage(nn.Module):
    """
    Realization of Swin Stage
    """
    def __init__(self, input_dim : int, output_dim : int, depth : int = 2):
        super().__init__()
        self.downsample = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.swin_blocks = nn.Sequential(
            *[SwinBlock(output_dim, output_dim) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.swin_blocks(x)
        return x

class SwinTransformer(nn.Module):
    """
    Realization of Swin Transformer
    """
    def __init__(self, num_classes : int, hidden_dim : int = 64):
        super().__init__()
        self.stage1 = SwinStage(3, hidden_dim, 2)
        self.stage2 = SwinStage(hidden_dim, 2*hidden_dim, 2)
        self.stage3 = SwinStage(2*hidden_dim, 4*hidden_dim, 6)
        self.conv_transpose1 = nn.ConvTranspose2d(4*hidden_dim, 2*hidden_dim, kernel_size=16, stride=16)
        self.conv_transpose2 = nn.ConvTranspose2d(2*hidden_dim, num_classes, kernel_size=16, stride=16)
        self.conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)


    def forward(self, x):
        # x : [B, height, width, channels]
        _, height, width, _ = x.shape
        x = x.permute(0, 3, 1, 2) # [B, channels, height, width]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv(x) # [B, num_classes, height', width']
        x = nn.functional.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 3, 1) # [B, height', width', num_classes]
        return x
