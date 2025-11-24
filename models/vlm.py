import torch
import torch.nn as nn
from PIL import Image

from .language_model import LanguageModel
from .projection_layer import VisionProjection
from .vision_encoder import VisionEncoder


class VLMChatbot(nn.Module):
    def __init__(
        self,
        vision_model="openai/clip-vit-large-patch14",
        language_model="lmsys/vicuna-7b-v1.5",
        load_in_8bit=False,
        load_in_4bit=False,
        checkpoint_path=None,
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(vision_model)
        self.language_model = LanguageModel(
            language_model,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            additional_special_tokens=["<image>", "</image>"],
        )
        self.projection = VisionProjection(
            vision_hidden_size=self.vision_encoder.hidden_size,
            llm_hidden_size=self.language_model.hidden_size,
        )

        self.image_token_id = self.language_model.tokenizer.convert_tokens_to_ids("<image>")

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward(self, images, text, labels=None):
        vision_features = self.vision_encoder(images)
        vision_embeddings = self.projection(vision_features)

        if isinstance(text, str):
            text_inputs = self.language_model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
                vision_embeddings.device
            )
        else:
            text_inputs = text

        combined_embeds, combined_mask = self.language_model.prepare_inputs_with_vision(text_inputs, vision_embeddings)

        outputs = self.language_model(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)

        return outputs

    @torch.no_grad()
    def chat(self, image_path, question, max_new_tokens=100, temperature=0.7):
        self.eval()

        image = Image.open(image_path).convert("RGB")
        vision_features = self.vision_encoder([image])
        vision_embeddings = self.projection(vision_features)

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        text_inputs = self.language_model.tokenizer(prompt, return_tensors="pt").to(vision_embeddings.device)

        combined_embeds, combined_mask = self.language_model.prepare_inputs_with_vision(text_inputs, vision_embeddings)

        output_ids = self.language_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.language_model.tokenizer.pad_token_id,
            eos_token_id=self.language_model.tokenizer.eos_token_id,
        )

        # Decode the full output
        full_output = self.language_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Try to extract the response after "ASSISTANT:"
        if "ASSISTANT:" in full_output:
            response = full_output.split("ASSISTANT:")[-1].strip()
        else:
            response = full_output.strip()

        # Clean up any remaining prompt text
        if "USER:" in response:
            response = response.split("USER:")[0].strip()

        return response

    def get_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "projection_state_dict": self.projection.state_dict(),
                "vision_encoder_state_dict": self.vision_encoder.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")

        if "projection_state_dict" in checkpoint:
            self.projection.load_state_dict(checkpoint["projection_state_dict"])
            print("Loaded projection layer weights")

        if "vision_encoder_state_dict" in checkpoint:
            self.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
            print("Loaded vision encoder weights")

        print("Checkpoint loaded successfully!")
