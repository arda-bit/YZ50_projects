from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12   # number of layers
    n_head: int = 12    # number of heads
    n_embd: int = 768   # embedding dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # not really a 'bias', more of causal attn mask, following the OpenAI/HF naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()    # b_sz, seq_len, n_embd
        # print(f'B: {B}, T: {T}, C: {C}')
        assert C == self.n_embd

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print(f'att.shape: {att.shape}, q.shape: {q.shape}, k.shape: {k.shape}')
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # concatenate all heads back together
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # shape (T,)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # print(f'GPT.forward(x).shape: {x.shape}')
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from HF"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask/buffer

        # init a HF transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                print(f'loaded {k} successfully')
            
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                print(f'loaded {k} successfully')
        return model

# attempt to auto detect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'using device: {device}')

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = GPT.from_pretrained('gpt2')
    print(f'model loaded successfully!')
    model.eval()
    model.to(device)    # move the entire model to GPU

    # prefix tokens
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a langauge model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (10,)
    tokens = tokens.unsqueeze(0) # (1, 10)
    tokens = tokens.to(device)

    # logits from our model
    with torch.no_grad():
        logits, _ = model(tokens)   # (B, T, vocab_size)

    # logits from HF
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    model_hf.eval()
    model_hf.to(device)

    with torch.no_grad():
        output_hf = model_hf(tokens)
        logits_hf = output_hf.logits   # (B, T, vocab_size)

    # compare the logits
    max_diff = (logits - logits_hf).abs().max().item()
    print(f"Max absolute difference: {max_diff:.8e}")
    print(f"torch.allclose(rtol=1e-4, atol=1e-4): {torch.allclose(logits, logits_hf, rtol=1e-4, atol=1e-4)}")


"""
loading weights from pretrained gpt: gpt2
Loading weights: 100%|█████████████████████████████████████████████████| 148/148 [00:00<00:00, 1700.54it/s]
loaded transformer.wte.weight successfully
loaded transformer.wpe.weight successfully
loaded transformer.h.0.ln_1.weight successfully
loaded transformer.h.0.ln_1.bias successfully
loaded transformer.h.0.attn.c_attn.weight successfully
loaded transformer.h.0.attn.c_attn.bias successfully
loaded transformer.h.0.attn.c_proj.weight successfully
loaded transformer.h.0.attn.c_proj.bias successfully
loaded transformer.h.0.ln_2.weight successfully
loaded transformer.h.0.ln_2.bias successfully
loaded transformer.h.0.mlp.c_fc.weight successfully
loaded transformer.h.0.mlp.c_fc.bias successfully
loaded transformer.h.0.mlp.c_proj.weight successfully
loaded transformer.h.0.mlp.c_proj.bias successfully
loaded transformer.h.1.ln_1.weight successfully
loaded transformer.h.1.ln_1.bias successfully
loaded transformer.h.1.attn.c_attn.weight successfully
loaded transformer.h.1.attn.c_attn.bias successfully
loaded transformer.h.1.attn.c_proj.weight successfully
loaded transformer.h.1.attn.c_proj.bias successfully
loaded transformer.h.1.ln_2.weight successfully
loaded transformer.h.1.ln_2.bias successfully
loaded transformer.h.1.mlp.c_fc.weight successfully
loaded transformer.h.1.mlp.c_fc.bias successfully
loaded transformer.h.1.mlp.c_proj.weight successfully
loaded transformer.h.1.mlp.c_proj.bias successfully
loaded transformer.h.2.ln_1.weight successfully
loaded transformer.h.2.ln_1.bias successfully
loaded transformer.h.2.attn.c_attn.weight successfully
loaded transformer.h.2.attn.c_attn.bias successfully
loaded transformer.h.2.attn.c_proj.weight successfully
loaded transformer.h.2.attn.c_proj.bias successfully
loaded transformer.h.2.ln_2.weight successfully
loaded transformer.h.2.ln_2.bias successfully
loaded transformer.h.2.mlp.c_fc.weight successfully
loaded transformer.h.2.mlp.c_fc.bias successfully
loaded transformer.h.2.mlp.c_proj.weight successfully
loaded transformer.h.2.mlp.c_proj.bias successfully
loaded transformer.h.3.ln_1.weight successfully
loaded transformer.h.3.ln_1.bias successfully
loaded transformer.h.3.attn.c_attn.weight successfully
loaded transformer.h.3.attn.c_attn.bias successfully
loaded transformer.h.3.attn.c_proj.weight successfully
loaded transformer.h.3.attn.c_proj.bias successfully
loaded transformer.h.3.ln_2.weight successfully
loaded transformer.h.3.ln_2.bias successfully
loaded transformer.h.3.mlp.c_fc.weight successfully
loaded transformer.h.3.mlp.c_fc.bias successfully
loaded transformer.h.3.mlp.c_proj.weight successfully
loaded transformer.h.3.mlp.c_proj.bias successfully
loaded transformer.h.4.ln_1.weight successfully
loaded transformer.h.4.ln_1.bias successfully
loaded transformer.h.4.attn.c_attn.weight successfully
loaded transformer.h.4.attn.c_attn.bias successfully
loaded transformer.h.4.attn.c_proj.weight successfully
loaded transformer.h.4.attn.c_proj.bias successfully
loaded transformer.h.4.ln_2.weight successfully
loaded transformer.h.4.ln_2.bias successfully
loaded transformer.h.4.mlp.c_fc.weight successfully
loaded transformer.h.4.mlp.c_fc.bias successfully
loaded transformer.h.4.mlp.c_proj.weight successfully
loaded transformer.h.4.mlp.c_proj.bias successfully
loaded transformer.h.5.ln_1.weight successfully
loaded transformer.h.5.ln_1.bias successfully
loaded transformer.h.5.attn.c_attn.weight successfully
loaded transformer.h.5.attn.c_attn.bias successfully
loaded transformer.h.5.attn.c_proj.weight successfully
loaded transformer.h.5.attn.c_proj.bias successfully
loaded transformer.h.5.ln_2.weight successfully
loaded transformer.h.5.ln_2.bias successfully
loaded transformer.h.5.mlp.c_fc.weight successfully
loaded transformer.h.5.mlp.c_fc.bias successfully
loaded transformer.h.5.mlp.c_proj.weight successfully
loaded transformer.h.5.mlp.c_proj.bias successfully
loaded transformer.h.6.ln_1.weight successfully
loaded transformer.h.6.ln_1.bias successfully
loaded transformer.h.6.attn.c_attn.weight successfully
loaded transformer.h.6.attn.c_attn.bias successfully
loaded transformer.h.6.attn.c_proj.weight successfully
loaded transformer.h.6.attn.c_proj.bias successfully
loaded transformer.h.6.ln_2.weight successfully
loaded transformer.h.6.ln_2.bias successfully
loaded transformer.h.6.mlp.c_fc.weight successfully
loaded transformer.h.6.mlp.c_fc.bias successfully
loaded transformer.h.6.mlp.c_proj.weight successfully
loaded transformer.h.6.mlp.c_proj.bias successfully
loaded transformer.h.7.ln_1.weight successfully
loaded transformer.h.7.ln_1.bias successfully
loaded transformer.h.7.attn.c_attn.weight successfully
loaded transformer.h.7.attn.c_attn.bias successfully
loaded transformer.h.7.attn.c_proj.weight successfully
loaded transformer.h.7.attn.c_proj.bias successfully
loaded transformer.h.7.ln_2.weight successfully
loaded transformer.h.7.ln_2.bias successfully
loaded transformer.h.7.mlp.c_fc.weight successfully
loaded transformer.h.7.mlp.c_fc.bias successfully
loaded transformer.h.7.mlp.c_proj.weight successfully
loaded transformer.h.7.mlp.c_proj.bias successfully
loaded transformer.h.8.ln_1.weight successfully
loaded transformer.h.8.ln_1.bias successfully
loaded transformer.h.8.attn.c_attn.weight successfully
loaded transformer.h.8.attn.c_attn.bias successfully
loaded transformer.h.8.attn.c_proj.weight successfully
loaded transformer.h.8.attn.c_proj.bias successfully
loaded transformer.h.8.ln_2.weight successfully
loaded transformer.h.8.ln_2.bias successfully
loaded transformer.h.8.mlp.c_fc.weight successfully
loaded transformer.h.8.mlp.c_fc.bias successfully
loaded transformer.h.8.mlp.c_proj.weight successfully
loaded transformer.h.8.mlp.c_proj.bias successfully
loaded transformer.h.9.ln_1.weight successfully
loaded transformer.h.9.ln_1.bias successfully
loaded transformer.h.9.attn.c_attn.weight successfully
loaded transformer.h.9.attn.c_attn.bias successfully
loaded transformer.h.9.attn.c_proj.weight successfully
loaded transformer.h.9.attn.c_proj.bias successfully
loaded transformer.h.9.ln_2.weight successfully
loaded transformer.h.9.ln_2.bias successfully
loaded transformer.h.9.mlp.c_fc.weight successfully
loaded transformer.h.9.mlp.c_fc.bias successfully
loaded transformer.h.9.mlp.c_proj.weight successfully
loaded transformer.h.9.mlp.c_proj.bias successfully
loaded transformer.h.10.ln_1.weight successfully
loaded transformer.h.10.ln_1.bias successfully
loaded transformer.h.10.attn.c_attn.weight successfully
loaded transformer.h.10.attn.c_attn.bias successfully
loaded transformer.h.10.attn.c_proj.weight successfully
loaded transformer.h.10.attn.c_proj.bias successfully
loaded transformer.h.10.ln_2.weight successfully
loaded transformer.h.10.ln_2.bias successfully
loaded transformer.h.10.mlp.c_fc.weight successfully
loaded transformer.h.10.mlp.c_fc.bias successfully
loaded transformer.h.10.mlp.c_proj.weight successfully
loaded transformer.h.10.mlp.c_proj.bias successfully
loaded transformer.h.11.ln_1.weight successfully
loaded transformer.h.11.ln_1.bias successfully
loaded transformer.h.11.attn.c_attn.weight successfully
loaded transformer.h.11.attn.c_attn.bias successfully
loaded transformer.h.11.attn.c_proj.weight successfully
loaded transformer.h.11.attn.c_proj.bias successfully
loaded transformer.h.11.ln_2.weight successfully
loaded transformer.h.11.ln_2.bias successfully
loaded transformer.h.11.mlp.c_fc.weight successfully
loaded transformer.h.11.mlp.c_fc.bias successfully
loaded transformer.h.11.mlp.c_proj.weight successfully
loaded transformer.h.11.mlp.c_proj.bias successfully
loaded transformer.ln_f.weight successfully
loaded transformer.ln_f.bias successfully
loaded lm_head.weight successfully
model loaded successfully!

Loading weights: 100%|████████████████████████████████████████| 148/148 [00:00<00:00, 1753.35it/s]
Max absolute difference: 6.10351562e-05
torch.allclose(rtol=1e-4, atol=1e-4): True
"""
