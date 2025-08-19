import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, features=[32, 64, 128, 256], dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        input_channels = in_channels
        for output_channels in features:
            self.layers.append(
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=input_channels,
                    out_channels=output_channels,
                    strides=2,
                    kernel_size=3,
                    dropout=dropout
                )
            )
            input_channels = output_channels

    def forward(self, x):
        features = []
        for block in self.layers:
            x = block(x)
            features.append(x)
        return features


# Lightweight Transformer Block
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        B, D, H, W, C = x.shape
        x_flat = x.view(B, -1, C)
        x_attn, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x = x_flat + x_attn
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, depths=[2, 2, 2], num_heads=[4, 4, 4], mlp_ratio=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            for _ in range(depths[i]):
                self.blocks.append(
                    SimpleTransformerBlock(
                        dim=embed_dim,
                        heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                    )
                )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x            


# Fusion Layer
class FusionLayer(nn.Module):
    def __init__(self, alpha_init=0.5, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init))

    def forward(self, cnn_feat, trans_feat):
        if cnn_feat.shape != trans_feat.shape:
            trans_feat = nn.functional.interpolate(trans_feat, size=cnn_feat.shape[2:], mode="trilinear", align_corners=False)
        return self.alpha * trans_feat + (1 - self.alpha) * cnn_feat


# Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=1, skip_connections=True):
        super().__init__()
        self.skip = skip_connections
        reversed_channels = in_channels_list[::-1]
        self.up_blocks = nn.ModuleList()

        for i in range(len(reversed_channels) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(reversed_channels[i], reversed_channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm3d(reversed_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(reversed_channels[-1], reversed_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm3d(reversed_channels[-1]),
            nn.ReLU(inplace=True)
        )    

        self.final_conv = nn.Conv3d(reversed_channels[-1], out_channels, kernel_size=1)

    def forward(self, features):
        x = features[-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if self.skip:
                skip = features[-(i + 2)]
                if skip.shape[2:] != x.shape[2:]:
                    skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="trilinear", align_corners=False)
                x = x + skip
        # ADDITIONAL upsample to get to full 128³
        x = self.final_upsample(x)
        return self.final_conv(x)
    
        

# NeuroSegNet Full Model
class NeuroSegNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        cnn_cfg = config["cnn_encoder"]
        trans_cfg = config["transformer"]
        fusion_cfg = config["fusion"]
        dec_cfg = config["decoder"]

        self.encoder = CNNEncoder(
            in_channels=model_cfg["in_channels"],
            features=cnn_cfg["features"],
            dropout=cnn_cfg["dropout"]
        )

        self.bridge = nn.Conv3d(cnn_cfg["features"][-1], trans_cfg["embed_dim"], kernel_size=1)
        self.transformer = TransformerEncoder(
            embed_dim=trans_cfg["embed_dim"],
            depths=trans_cfg["depths"],
            num_heads=trans_cfg["num_heads"],
            mlp_ratio=trans_cfg["mlp_ratio"],
            dropout=trans_cfg["dropout"],
            attention_dropout=trans_cfg["attention_dropout"]
        )

        self.back_proj = nn.Conv3d(trans_cfg["embed_dim"], cnn_cfg["features"][-1], kernel_size=1)
        self.fusion = FusionLayer(learnable=fusion_cfg["learnable_alpha"])
        self.decoder = Decoder(
            in_channels_list=cnn_cfg["features"],
            out_channels=model_cfg["out_channels"],
            skip_connections=dec_cfg["skip_connections"]
        )

    def forward(self, x):
        cnn_feats = self.encoder(x)
        bridge_out = self.bridge(cnn_feats[-1])
        trans_out = self.transformer(bridge_out)
        trans_back = self.back_proj(trans_out)
        fused = self.fusion(cnn_feats[-1], trans_back)
        cnn_feats[-1] = fused
        out = self.decoder(cnn_feats)
        return out
