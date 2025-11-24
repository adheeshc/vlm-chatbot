import unittest
import warnings

import torch

from models.language_model import LanguageModel
from tests.test_utils import BaseTest, create_dummy_tensor, create_dummy_text_inputs


class TestLanguageModel(BaseTest):
    """Test suite for LanguageModel"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        # Use a smaller model for testing to save resources
        cls.model_name = "gpt2"  # Lightweight model for testing
        cls.vision_seq_length = 256
        cls.text_seq_length = 10

    def test_tokenizer_initialization(self):
        """Test that tokenizer is properly initialized"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        self.assertIsNotNone(model.tokenizer)
        self.assertIsNotNone(model.tokenizer.pad_token, "Pad token should be set")

    def test_model_initialization(self):
        """Test that language model initializes correctly"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        self.assertIsNotNone(model.model)
        self.assertGreater(model.hidden_size, 0, "Hidden size should be positive")

    def test_prepare_inputs_with_vision(self):
        """Test combining vision and text embeddings"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        batch_size = 2
        vision_embeddings = create_dummy_tensor(batch_size, self.vision_seq_length, model.hidden_size)

        text_inputs = create_dummy_text_inputs(batch_size=batch_size, seq_length=self.text_seq_length)

        combined_embeddings, combined_attention_mask = model.prepare_inputs_with_vision(text_inputs, vision_embeddings)

        # Check combined embeddings shape
        expected_seq_length = self.vision_seq_length + self.text_seq_length
        expected_shape = (batch_size, expected_seq_length, model.hidden_size)
        self.assertEqual(
            tuple(combined_embeddings.shape),
            expected_shape,
            f"Expected shape {expected_shape}, got {tuple(combined_embeddings.shape)}",
        )

        # Check attention mask shape
        expected_mask_shape = (batch_size, expected_seq_length)
        self.assertEqual(
            tuple(combined_attention_mask.shape),
            expected_mask_shape,
            f"Expected shape {expected_mask_shape}, got {tuple(combined_attention_mask.shape)}",
        )

        # Check that all attention values are 1 (all tokens attended to)
        self.assertTrue(torch.all(combined_attention_mask == 1), "All positions should be attended to")

    def test_forward_pass(self):
        """Test forward pass through the model"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        batch_size = 1
        seq_length = 20

        inputs_embeds = create_dummy_tensor(batch_size, seq_length, model.hidden_size)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        with torch.no_grad():
            outputs = model(inputs_embeds, attention_mask)

        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "logits"), "Output should have logits attribute")

    def test_generate(self):
        """Test text generation"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        batch_size = 1
        seq_length = 10

        inputs_embeds = create_dummy_tensor(batch_size, seq_length, model.hidden_size)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        max_new_tokens = 5
        outputs = model.generate(inputs_embeds, attention_mask, max_new_tokens=max_new_tokens)

        self.assertIsNotNone(outputs)

        # Handle both tensor output and GenerateOutput objects
        if isinstance(outputs, torch.Tensor):
            output_tensor = outputs
        else:
            output_tensor = outputs.sequences

        self.assertIsInstance(output_tensor, torch.Tensor)
        self.assertEqual(len(output_tensor.shape), 2, "Output should be 2D (batch_size, sequence_length)")

    def test_vision_text_combined_shape(self):
        """Test that vision and text embeddings combine correctly"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        batch_size = 1
        vision_seq = 100
        text_seq = 50

        vision_embeddings = create_dummy_tensor(batch_size, vision_seq, model.hidden_size)
        text_inputs = create_dummy_text_inputs(batch_size=batch_size, seq_length=text_seq)

        combined_embeddings, combined_attention_mask = model.prepare_inputs_with_vision(text_inputs, vision_embeddings)

        # First part should be vision embeddings
        vision_part = combined_embeddings[:, :vision_seq, :]
        self.assertEqual(vision_part.shape[1], vision_seq)

        # Total sequence length should be vision + text
        self.assertEqual(combined_embeddings.shape[1], vision_seq + text_seq)

    def test_hidden_size_consistency(self):
        """Test that hidden size is consistent across components"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LanguageModel(model_name=self.model_name, load_in_8bit=False)

        self.assertEqual(model.hidden_size, model.model.config.hidden_size, "Hidden size should match model config")


if __name__ == "__main__":
    unittest.main()
