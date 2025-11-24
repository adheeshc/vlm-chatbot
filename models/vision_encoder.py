import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel


class VisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            images = self.processor(images=images, return_tensors="pt")["pixel_values"]
            images = images.to(self.model.device)

        outputs = self.model(images, output_hidden_states=True)
        vision_features = outputs.last_hidden_state[:, 1:, :]

        return vision_features
