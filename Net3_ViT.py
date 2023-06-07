import torch
import torch.nn as nn
import vit_pytorch as vit

from homework3 import DiceLoss


class ViTAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vit.ViT(
            image_size=256,
            patch_size=16,
            num_classes=1000,
            dim=1024,
            depth=12,
            heads=16,
            mlp_dim=4096
        )
        self.to_patch = self.encoder.to_patch_embedding[:1]
        self.patch_to_emb = self.encoder.to_patch_embedding[1:]
        self.decoder = vit.vit.Transformer(
            dim=1024,
            depth=12,
            heads=16,
            dim_head=64,
            mlp_dim=4096
        )
        self.to_pixels = nn.Sequential(
            nn.Linear(1024, 256),
            # nn.Sigmoid()
        )

    def forward(self, x):
        patches = self.to_patch(x)

        tokens = self.patch_to_emb(patches)
        num_patches = 256
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        encoded_tokens = self.encoder.transformer(tokens)
        decoded_tokens = self.decoder(encoded_tokens)

        z = self.to_pixels(decoded_tokens)
        return z

    def loss(self, x, y, costFunc):
        y = self.to_patch(y)
        return costFunc(x, y)
        # return DiceLoss.calc(x, y)
