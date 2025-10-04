import sys
from pathlib import Path

# Make minGPT accessible
ROOT = Path(__file__).resolve().parents[2]  # /home/.../minGPT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# my_HR_project code
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from functools import lru_cache
import traceback
set_seed(3407)

def generate(use_mingpt, model_type, prompt, device, model, steps=20, do_sample=True):
    
    empty_prompt = False
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            empty_prompt = True
            pass
        else:
            x = tokenizer(prompt).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '': 
            empty_prompt = True
            pass
        else:
            encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
            x = encoded_input['input_ids']
    
    
    
    if not empty_prompt:
        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

        start = x.size(1) # the generation starts after the prompt

        out = tokenizer.decode(y[0, start:].cpu())
    else:
        out = "no prompt given, plese retry following the instructions above"
    return out


@lru_cache(maxsize=1)
def _load_model_cached(use_mingpt: bool, model_type: str):
    """Load and cache the model to avoid re-initialization on every call."""
    if use_mingpt:
        model = GPT.from_pretrained(model_type)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_type)
        model.config.pad_token_id = model.config.eos_token_id  # suppress a warning
    device = 'cpu'
    model.to(device)
    model.eval()
    return model, device


def get_reply(user_message):
    use_mingpt = True  # use minGPT or huggingface/transformers model?
    model_type = 'gpt2'

    # Load model once and reuse across calls; also guards CUDA I/O issues
    model, device = _load_model_cached(use_mingpt, model_type)

    reply = generate(
        use_mingpt=use_mingpt,
        model_type=model_type,
        prompt=user_message,
        device="cpu",
        model=model,
        steps=20,
        do_sample=True,
    )

    return reply



# DEBUG
print("reply: ", get_reply("hello, I am"))
