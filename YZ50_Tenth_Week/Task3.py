import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from Task2 import (
    GPT
)

# attempt to auto detect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'using device: {device}')

if __name__ == '__main__':
    
    num_return_sequences = 5
    max_length = 30

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = GPT.from_pretrained('gpt2')
    print(f'model loaded successfully!')
    model.eval()
    model.to('cuda')    # move the entire model to GPU

    en_string = "Hello, I'm a large langauge model,"
    tr_string = "Merhaba, ben bir büyük dil modeliyim,"

    enc = tiktoken.get_encoding('gpt2')
    en_tokens = enc.encode(en_string)
    tr_tokens = enc.encode(tr_string)

    print(f'en_string is {len(en_tokens)} tokens long')
    print(f'tr_string is {len(tr_tokens)} tokens long')

    en_tokens = torch.tensor(en_tokens, dtype=torch.long) # (10,)
    en_tokens = en_tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 10)

    tr_tokens = torch.tensor(tr_tokens, dtype=torch.long) # (11,)
    tr_tokens = tr_tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 11)
    
    # since these seq not equal length on dim=1, need to prepend a pad to the shorter one
    pad_id = enc.eot_token # token id: 50256 (<|endoftext|>)
    diff = tr_tokens.size(1) - en_tokens.size(1)
    if diff > 0:
        en_tokens = F.pad(en_tokens, (diff, 0), value=pad_id)
    elif diff < 0:
        tr_tokens = F.pad(tr_tokens, (abs(diff), 0), value=pad_id)

    tokens = torch.cat((en_tokens, tr_tokens), dim=0) # (2*B, max_T)
    x = tokens.to('cuda')

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)   # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]   # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (HF pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)   # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)   # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(x.size(0)):
        out_tokens = x[i, :max_length].tolist()
        # Filter out pad tokens (<|endoftext|>) before decoding
        out_tokens = [t for t in out_tokens if t != pad_id]
        decoded = enc.decode(out_tokens)
        print(f'> {decoded}')

"""
Responses with no batched generation of en + tr

> Hello, I'm a langauge model, I'm from the Netherlands and the game-show of course, was "Grimoire of War
> Hello, I'm a langauge model, and i'm a big fan of H.P. Lovecraft's "Lord of the Rings" trilogy
> Hello, I'm a langauge model, and I write the syntax."

"Okay, so it sounds like you're in the same
> Hello, I'm a langauge model, you won't understand. I am an expert photographer. I worked with all kinds of subjects."

> Hello, I'm a langauge model, a kind of anthropologist with a love for literature on the side. To me, writing about things
"""
"""
Responses with batched generation of en + tr (B=10, 5 en + 5 tr)

en_string is 11 tokens long
tr_string is 16 tokens long
> Hello, I'm a large langauge model, and I think it is the only way to see all the features with
> Hello, I'm a large langauge model, so as a result, I am interested in learning whether other people like
> Hello, I'm a large langauge model, I'm based in Montreal, Canada, and as such I work for
> Hello, I'm a large langauge model, and there's nothing like a new concept for writing a simple example and
> Hello, I'm a large langauge model, which means that we're working on multiple layers of objects. So,
> Merhaba, ben bir büyük dil modeliyim, dum kez dukün'an aye kü
> Merhaba, ben bir büyük dil modeliyim, de veldig, and de ye yük ben o
> Merhaba, ben bir büyük dil modeliyim, uxulü hosu werde pátaç
> Merhaba, ben bir büyük dil modeliyim, vosükzük

Lazak, z
> Merhaba, ben bir büyük dil modeliyim, is my project of choice. The main goal of mine is to improve
"""