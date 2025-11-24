import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LanguageModel(nn.Module):
    def __init__(
        self, model_name="lmsys/vicuna-7b-v1.5", load_in_8bit=False, load_in_4bit=False, additional_special_tokens=None
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if additional_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quantization_config, device_map={"": 0}, low_cpu_mem_usage=True
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="balanced",
                max_memory={0: "7GiB", "cpu": "20GiB"},
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if additional_special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.hidden_size = self.model.config.hidden_size

    def prepare_inputs_with_vision(self, text_inputs, vision_embeddings):
        input_ids = text_inputs["input_ids"]
        embed_device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(embed_device)

        text_embeddings = self.model.get_input_embeddings()(input_ids)
        vision_embeddings = vision_embeddings.to(device=text_embeddings.device, dtype=text_embeddings.dtype)
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)

        batch_size = text_embeddings.shape[0]

        vision_attention = torch.ones(
            batch_size,
            vision_embeddings.shape[1],
            dtype=text_inputs["attention_mask"].dtype,
            device=text_embeddings.device,
        )

        text_attention_mask = text_inputs["attention_mask"].to(text_embeddings.device)

        combined_attention_mask = torch.cat([vision_attention, text_attention_mask], dim=1)

        return combined_embeddings, combined_attention_mask

    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True
        )
        return outputs

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens=100, **kwargs):
        attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **kwargs
        )
        return outputs
