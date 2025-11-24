import torch.nn as nn


class VisionProjection(nn.Module):
    def __init__(self, vision_hidden_size=1024, llm_hidden_size=4096):
        super().__init__()
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)

    def forward(self, vision_features):
        return self.projection(vision_features)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
