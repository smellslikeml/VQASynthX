# Adapted from SOURCE: imdb_gen/classifier.py
# This file defines the custom model used for PITA classification.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class CustomLlamaForClassification(nn.Module):
    def __init__(self, model_id, use_bias=True):
        super(CustomLlamaForClassification, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.config = self.model.config
        self.use_bias = use_bias

        # PITA adds a linear layer and optionally a bias to the model's output
        self.value_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        if self.use_bias:
            self.value_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]
        logits = self.value_head(hidden_states)

        if self.use_bias:
            logits += self.value_bias

        return logits.squeeze(-1)
