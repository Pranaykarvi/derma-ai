import torch
import torch.nn as nn
import timm
from pathlib import Path

# Define the same ViTEffNetFusion model as used during training
class ViTEffNetFusion(nn.Module):
    def __init__(self, num_classes):
        super(ViTEffNetFusion, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.effnet = timm.create_model('tf_efficientnet_b0', pretrained=False)

        self.vit.head = nn.Identity()
        self.effnet.classifier = nn.Identity()

        vit_feat_dim = self.vit.num_features
        eff_feat_dim = self.effnet.num_features

        self.fc = nn.Sequential(
            nn.Linear(vit_feat_dim + eff_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)
        eff_feat = self.effnet(x)
        x = torch.cat([vit_feat, eff_feat], dim=1)
        return self.fc(x)

# Load the model
def load_model(model_path: str = "backend/model/model_fold1.pth", num_classes: int = 7):

    model = ViTEffNetFusion(num_classes=num_classes)

    if Path(model_path).suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    model.load_state_dict(state_dict)
    model.eval()
    return model
