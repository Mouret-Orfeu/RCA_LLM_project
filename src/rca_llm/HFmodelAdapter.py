"""
Encapsulation class for HuggingFace models to be trained with my trainer class
"""

from transformers import AutoTokenizer
from transformers.trainer import get_parameter_names
import torch, torch.nn as nn

class HFmodelAdapter(nn.Module):
    def __init__(self, hf_model, model_type):
        super().__init__()
        self.hf_model = hf_model
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        # Ensure pad_token_id exists for generation warnings
        # You give the config a pading token (here the eos token) if it doesn't have one
        # The token used for padding does not matter since we will mask out the loss on those tokens
        if self.hf_model.config.pad_token_id is None:
            self.hf_model.config.pad_token_id = self.hf_model.config.eos_token_id
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # here, idx and target are the x and y returned by the dataloader in trainer.py
    def forward(self, idx, targets=None):
        out = self.hf_model(input_ids=idx, labels=targets)
        # HF returns a ModelOutput with .logits and .loss
        # getattr(obj, name, default) tries to read attribute name from obj. 
        # If it exists, you get its value; if it doesn’t exist, you get default instead of raising an AttributeError.
        # HF models have a default parameter: return_dict=True, if it is set to False by mistake, then the getattr will avoid a crash by AttributeError
        return out.logits, getattr(out, "loss", None)

    # def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None): # try to replace with the generate function in generate.ipynb
    #     gen_kwargs = dict(
    #         max_new_tokens=max_new_tokens,
    #         do_sample=do_sample,
    #         temperature=temperature,
    #         pad_token_id=hf_model.config.pad_token_id,
    #     )
    #     if top_k is not None:
    #         gen_kwargs["top_k"] = top_k
    #     return hf_model.generate(input_ids=idx, **gen_kwargs)
    
    def generate_from_prompt(
        self,
        prompt,
        device,
        max_new_tokens=20,
        do_sample=True,
        temperature=1.0,
        top_k=None,
        return_new_text_only=True,
        skip_special_tokens=True,
    ):
        if not prompt:
            return "no prompt given, please provide a non-empty prompt"
    
        
        encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = encoded_input['input_ids'].to(device)

        attention_mask = encoded_input.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        y = self.hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.hf_model.config.pad_token_id,
            eos_token_id=self.hf_model.config.eos_token_id
        )

        if return_new_text_only:
            new_ids = y[0, input_ids.shape[1]:].tolist()
            return self.tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(y[0].tolist(), skip_special_tokens=skip_special_tokens)

    # def configure_optimizers(self, train_config): # try to replace withe the configure_optimizers function in model.py
    #     # Similar to minGPT: decay on Linear/Conv1D weights, no decay on bias/LayerNorm/Embedding
    #     decay, no_decay = [], []
    #     whitelist = (torch.nn.Linear,)
    #     blacklist = (torch.nn.LayerNorm, torch.nn.Embedding)
    #     # HF GPT2 also uses a custom Conv1D; include it if present
    #     try:
    #         from transformers.pytorch_utils import Conv1D
    #         whitelist = (torch.nn.Linear, Conv1D)
    #     except Exception:
    #         pass

    #     for name, module in self.named_modules():
    #         for pn, p in module.named_parameters(recurse=False):
    #             if not p.requires_grad:
    #                 continue
    #             full = f"{name}.{pn}" if name else pn
    #             if pn.endswith("bias") or isinstance(module, blacklist):
    #                 no_decay.append(p)
    #             elif pn.endswith("weight") and isinstance(module, whitelist):
    #                 decay.append(p)
    #             else:
    #                 # fallback: treat as no_decay to be safe
    #                 no_decay.append(p)

    #     optim_groups = [
    #         {"params": decay, "weight_decay": train_config.weight_decay},
    #         {"params": no_decay, "weight_decay": 0.0},
    #     ]
    #     return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    

    def configure_optimizers(self, train_config):
        decay_names = get_parameter_names(self, [torch.nn.LayerNorm])
        decay_names = [n for n in decay_names if not n.endswith(".bias")]

        decay_params, nodecay_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay_params if n in decay_names else nodecay_params).append(p)

        groups = [
            {"params": decay_params,   "weight_decay": train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=train_config.learning_rate, betas=train_config.betas)