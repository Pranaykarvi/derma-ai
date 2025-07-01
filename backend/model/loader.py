import torch
import torch.nn as nn
import timm
from pathlib import Path

# 🔧 Fusion model combining ViT and EfficientNet
class ViTEffNetFusion(nn.Module):
    def __init__(self, num_classes: int):
        super(ViTEffNetFusion, self).__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.effnet = timm.create_model('tf_efficientnet_b0', pretrained=False)

        self.vit.head = nn.Identity()
        self.effnet.classifier = nn.Identity()

        vit_feat_dim = self.vit.num_features
        eff_feat_dim = self.effnet.num_features

        self.fc = nn.Sequential(
            nn.Linear(vit_feat_dim + eff_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)
        eff_feat = self.effnet(x)
        x = torch.cat([vit_feat, eff_feat], dim=1)
        return self.fc(x)


# ✅ Load model with state_dict
def load_model(model_path: str = "backend/model/model_fold1.pth", num_classes: int = 7):
    model = ViTEffNetFusion(num_classes=num_classes)

    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    model.to("cpu")  # Force CPU mode for low-RAM deployment
    return model
