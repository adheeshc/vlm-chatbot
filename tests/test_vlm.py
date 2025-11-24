import os
import tempfile
import unittest

import torch
from PIL import Image

from models.vlm import VLMChatbot
from tests.test_utils import BaseTest


class TestVLMChatbot(BaseTest):
    @classmethod
    def setUpClass(cls):
        """Set up model once for all tests."""
        print("\nInitializing VLMChatbot model with 4-bit quantization...")
        cls.model = VLMChatbot(load_in_4bit=True)

    def test_vision_encoder(self):
        """Test vision encoder processes images correctly."""
        image = Image.new("RGB", (224, 224), color="red")
        features = self.model.vision_encoder([image])

        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], 256)
        self.assertEqual(features.shape[2], self.model.vision_encoder.hidden_size)

    def test_projection_layer(self):
        """Test projection layer transforms features correctly."""
        vision_features = torch.randn(1, 256, 1024)
        projected = self.model.projection(vision_features)

        self.assertEqual(projected.shape, (1, 256, self.model.language_model.hidden_size))

    def test_end_to_end_inference(self):
        """Test end-to-end inference through chat method."""
        image = Image.new("RGB", (224, 224), color="blue")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            image.save(temp_path)
            response = self.model.chat(temp_path, "What color is this?", max_new_tokens=50)
            self.assertIsInstance(response, str)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parameter_counts(self):
        """Test parameter counting returns valid numbers."""
        trainable, total = self.model.get_trainable_parameters()

        print(f"Trainable: {trainable:,} / Total: {total:,}")
        print(f"Trainable ratio: {trainable/total*100:.2f}%")
        self.assertGreater(trainable, 0, "Trainable parameters should be greater than 0")
        self.assertGreater(total, 0, "Total parameters should be greater than 0")
        self.assertLessEqual(trainable, total, "Trainable parameters should not exceed total")
        self.assertGreater(total, 3_000_000_000, "Total parameters should be over 3 billion for quantized 7B model")


if __name__ == "__main__":
    unittest.main()
