import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin

class ViTLateFusion(nn.Module, PyTorchModelHubMixin):
    """
    Implements a late fusion Vision Transformer (ViT) model that processes
    three input views with shared weights, concatenates their features, and
    classifies the fused representation using a multi-layer perceptron.
    Supports both timm and HuggingFace ViT backends.
    """

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name

        if self.model_name in ["timm/vit_large_patch32_224.orig_in21k", "brunoasm/vit_large_patch32_224"]:
            import timm
            self.vit = timm.create_model(self.model_name, pretrained=True, num_classes=0) # removes their classification head, gives you raw features

            self.hidden_size = self.vit.num_features
            self.config = None  # timm models have no .config
            print("loaded timm model", model_name)
        else:
            self.vit = AutoModel.from_pretrained(self.model_name)
            self.config = self.vit.config
            self.hidden_size = self.vit.config.hidden_size


        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_labels)
        )

        self.num_labels = num_labels

    def forward(self, pixel_values_xy, pixel_values_yz, pixel_values_xz, labels=None):
        if self.model_name in ["timm/vit_large_patch32_224.orig_in21k", "brunoasm/vit_large_patch32_224"]:
            feat_xy = self.vit(pixel_values_xy)
            feat_yz = self.vit(pixel_values_yz)
            feat_xz = self.vit(pixel_values_xz)
        else:
            feat_xy = self.vit(pixel_values_xy).pooler_output
            feat_yz = self.vit(pixel_values_yz).pooler_output
            feat_xz = self.vit(pixel_values_xz).pooler_output

        # Fuse features from the 3 views
        fused_features = torch.cat((feat_xy, feat_yz, feat_xz), dim=1)

        # Classify
        logits = self.classifier(fused_features)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.01)(logits, labels)
        return {"loss": loss, "logits": logits, "features": fused_features}
