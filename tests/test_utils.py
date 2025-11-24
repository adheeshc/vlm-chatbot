import unittest
import torch
from PIL import Image


class BaseTest(unittest.TestCase):
    """Base test class with common setup and teardown"""

    def setUp(self):
        """Print test name before each test"""
        print(f"\nRunning: {self._testMethodName}")

    def tearDown(self):
        """Print pass status after each test"""
        print(f"SUCCESS: {self._testMethodName}")


def get_test_image_path():
    """Get path to test image"""
    return "data/test_images/dog.jpg"


def load_test_image():
    """Load a test image"""
    return Image.open(get_test_image_path())


def create_dummy_tensor(*shape, device='cpu', dtype=torch.float32):
    """Create a dummy tensor with given shape for testing"""
    return torch.randn(*shape, device=device, dtype=dtype)


def create_dummy_text_inputs(batch_size=1, seq_length=10):
    """Create dummy text inputs for testing language models"""
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones((batch_size, seq_length), dtype=torch.long)
    }
