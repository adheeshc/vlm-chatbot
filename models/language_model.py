import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LanguageModel(nn.Module):
    def __init__(self, model_name="lmsys/vicuna-7b-v1.5", load_in_8bit=False):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.hidden_size = self.model.config.hidden_size
        
    def prepare_inputs_with_vision(self, text_inputs, vision_embeddings):
        input_ids = text_inputs['input_ids']
        text_embeddings = self.model.get_input_embeddings()(input_ids)
        
        batch_size = text_embeddings.shape[0]
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
        
        vision_attention = torch.ones(
            batch_size, vision_embeddings.shape[1],
            dtype=text_inputs['attention_mask'].dtype,
            device=text_inputs['attention_mask'].device
        )
        combined_attention_mask = torch.cat([
            vision_attention,
            text_inputs['attention_mask']
        ], dim=1)
        
        return combined_embeddings, combined_attention_mask
    
    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, max_new_tokens=100, **kwargs):
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        return outputs