import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForMaskedLM


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze base weights
        for param in self.parameters():
            param.requires_grad = False
        for param in self.lora_A.parameters():
            param.requires_grad = True
        for param in self.lora_B.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.to(self.lora_A.weight.device)
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
    

class LoRARobertaMLM(nn.Module):
    def __init__(self, base_model_name="roberta-base", r=8, alpha=32):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained(base_model_name)
        self.inject_lora(r, alpha)

    def inject_lora(self, r, alpha):
      self.lora_modules = nn.ModuleList()  # Register all LoRA layers

      for name, module in self.model.named_modules():
          if isinstance(module, nn.Linear) and ("query" in name or "value" in name):
              in_dim = module.in_features
              out_dim = module.out_features
              lora = LoRALayer(in_dim, out_dim, r=r, alpha=alpha)
              self.lora_modules.append(lora)  # Register it so .to(device) works

              original_forward = module.forward
              module.forward = self._wrap_forward(original_forward, lora)
    def _wrap_forward(self, original_forward, lora_module):
        def wrapped(x):
            return original_forward(x) + lora_module(x)
        return wrapped

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)