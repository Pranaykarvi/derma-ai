import torch
import torch.nn as nn
import timm


class ViTEffNetFusion(nn.Module):
    def __init__(self, num_classes: int):
        super(ViTEffNetFusion, self).__init__()

        # Load Vision Transformer (ViT)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()  # Remove classification head

        # Load EfficientNet
        self.effnet = timm.create_model("tf_efficientnet_b0", pretrained=True)
        self.effnet.classifier = nn.Identity()  # Remove classification head

        # Fusion layer
        self.fc = nn.Linear(768 + 1280, num_classes)  # ViT output (768) + EffNet (1280)

    def forward(self, x):
        vit_feat = self.vit(x)      # Shape: [B, 768]
        eff_feat = self.effnet(x)   # Shape: [B, 1280]
        fused = torch.cat([vit_feat, eff_feat], dim=1)  # Shape: [B, 2048]
        output = self.fc(fused)     # Shape: [B, num_classes]
        return output
