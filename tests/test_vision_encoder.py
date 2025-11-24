import unittest
from models.vision_encoder import VisionEncoder
import torch
from tests.test_utils import BaseTest, load_test_image


class TestVisionEncoder(BaseTest):
    """Test suite for VisionEncoder"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        cls.encoder = VisionEncoder()
        cls.encoder.eval()

    def test_encoder_initialization(self):
        """Test that VisionEncoder initializes correctly"""
        self.assertIsNotNone(self.encoder)
        self.assertIsInstance(self.encoder, VisionEncoder)
        self.assertGreater(self.encoder.hidden_size, 0,
                          "Hidden size should be positive")

    def test_image_feature_extraction(self):
        """Test that features can be extracted from an image"""
        img = load_test_image()

        with torch.no_grad():
            features = self.encoder([img])

        self.assertIsNotNone(features)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features.shape), 3,
                        f"Expected 3D tensor, got shape: {features.shape}")
        self.assertTrue(torch.isfinite(features).all(),
                       "Tensor contains non-finite values")

        print(f"Vision features shape: {features.shape}")

    def test_feature_dimensions(self):
        """Test that feature dimensions are consistent"""
        img = load_test_image()

        with torch.no_grad():
            features = self.encoder([img])

        # Verify batch dimension matches input (single image)
        self.assertEqual(features.shape[0], 1,
                        "Batch size should be 1 for single image input")

        # Verify number of patches/tokens is positive
        self.assertGreater(features.shape[1], 0,
                          "Number of patches should be positive")

        # Verify feature dimension matches hidden size
        self.assertEqual(features.shape[2], self.encoder.hidden_size,
                        f"Feature dimension should match hidden_size: {self.encoder.hidden_size}")

    def test_multiple_images(self):
        """Test that encoder handles multiple images in batch"""
        img1 = load_test_image()
        img2 = load_test_image()

        with torch.no_grad():
            features = self.encoder([img1, img2])

        self.assertEqual(features.shape[0], 2,
                        "Batch size should be 2 for two images")
        self.assertTrue(torch.isfinite(features).all(),
                       "Tensor contains non-finite values")

    def test_tensor_input(self):
        """Test that encoder can accept tensor input"""
        img = load_test_image()

        # First get processed tensor
        with torch.no_grad():
            processed = self.encoder.processor(images=[img], return_tensors="pt")["pixel_values"]
            features = self.encoder(processed)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features.shape), 3)
        self.assertTrue(torch.isfinite(features).all(),
                       "Tensor contains non-finite values")

    def test_frozen_parameters(self):
        """Test that vision model parameters are frozen"""
        for param in self.encoder.model.parameters():
            self.assertFalse(param.requires_grad,
                           "Vision model parameters should be frozen")


if __name__ == '__main__':
    unittest.main()
