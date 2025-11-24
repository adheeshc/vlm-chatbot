import unittest
from models.projection_layer import VisionProjection
import torch
from tests.test_utils import BaseTest, create_dummy_tensor


class TestVisionProjection(BaseTest):
    """Test suite for VisionProjection"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        cls.vision_hidden_size = 1024
        cls.llm_hidden_size = 4096
        cls.projection = VisionProjection(
            vision_hidden_size=cls.vision_hidden_size,
            llm_hidden_size=cls.llm_hidden_size
        )

    def test_projection_initialization(self):
        """Test that VisionProjection initializes correctly"""
        self.assertIsNotNone(self.projection)
        self.assertIsInstance(self.projection, VisionProjection)

    def test_projection_layer_exists(self):
        """Test that projection layer is properly created"""
        self.assertIsNotNone(self.projection.projection)
        self.assertEqual(self.projection.projection.in_features, self.vision_hidden_size)
        self.assertEqual(self.projection.projection.out_features, self.llm_hidden_size)

    def test_weight_initialization(self):
        """Test that weights are properly initialized"""
        weights = self.projection.projection.weight
        bias = self.projection.projection.bias

        # Check that weights are not all zeros
        self.assertFalse(torch.all(weights == 0), "Weights should not all be zero")

        # Check that bias is initialized to zero
        if bias is not None:
            self.assertTrue(torch.all(bias == 0), "Bias should be initialized to zero")

    def test_forward_pass(self):
        """Test forward pass with dummy vision features"""
        batch_size = 2
        num_patches = 256

        # Create dummy vision features
        vision_features = create_dummy_tensor(
            batch_size, num_patches, self.vision_hidden_size
        )

        # Project features
        projected = self.projection(vision_features)

        # Verify output shape
        expected_shape = (batch_size, num_patches, self.llm_hidden_size)
        self.assertEqual(tuple(projected.shape), expected_shape,
                        f"Expected shape {expected_shape}, got {tuple(projected.shape)}")

    def test_output_finite(self):
        """Test that projection output contains finite values"""
        vision_features = create_dummy_tensor(1, 256, self.vision_hidden_size)

        projected = self.projection(vision_features)

        self.assertTrue(torch.isfinite(projected).all(),
                       "Tensor contains non-finite values")

    def test_parameter_count(self):
        """Test that parameter count is calculated correctly"""
        num_params = self.projection.get_num_parameters()

        # Calculate expected parameters (weight + bias)
        expected_params = self.vision_hidden_size * self.llm_hidden_size + self.llm_hidden_size

        self.assertEqual(num_params, expected_params,
                        f"Expected {expected_params} parameters, got {num_params}")

    def test_different_dimensions(self):
        """Test projection with different input/output dimensions"""
        vision_size = 512
        llm_size = 2048

        proj = VisionProjection(vision_hidden_size=vision_size, llm_hidden_size=llm_size)

        vision_features = create_dummy_tensor(1, 100, vision_size)
        projected = proj(vision_features)

        expected_shape = (1, 100, llm_size)
        self.assertEqual(tuple(projected.shape), expected_shape,
                        f"Expected shape {expected_shape}, got {tuple(projected.shape)}")

    def test_gradient_flow(self):
        """Test that gradients can flow through projection layer"""
        self.projection.train()

        vision_features = create_dummy_tensor(1, 10, self.vision_hidden_size)
        vision_features.requires_grad = True

        projected = self.projection(vision_features)
        loss = projected.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(vision_features.grad, "Gradients should flow to input")
        self.assertIsNotNone(self.projection.projection.weight.grad,
                           "Gradients should flow to weights")


if __name__ == '__main__':
    unittest.main()
