import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from GPT2_model import (
    GPT,
    GPTConfig,
)

# attempt to auto detect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding('gpt2')

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = os.path.join(os.path.dirname(__file__), 'edu_fineweb10B')
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'
        print(f'found {len(shards)} shards for split {split}')
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)   # inputs
        y = (buf[1:]).view(B, T)    # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# data loader parameters
total_batch_size = 524288   # 2**19, ~0.5M, in number of tokens
B = 8  # micro batch size
T = 1024    # sequence length
assert total_batch_size % (B * T) == 0, 'make sure total_batch_size is divisible by B * T'
grad_accum_steps = total_batch_size // (B * T)

print(f'total desired batch size: {total_batch_size}')
print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')

# cosine decay with warmup
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 10
max_steps = 50
eval_interval = 10

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it+1) / warmup_iters
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_val_loss(model, val_loader, device, val_loss_steps=20):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    val_loss = val_loss_accum.item()
    print(f'validation loss: {val_loss:.4f}')
    return val_loss


def generate_samples(
    model,
    enc,
    device,
    prompt="Hello, I'm a language model,",
    num_return_sequences=4,
    max_length=32,
):
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    # a better way is to do the generation with the original model and not recompile 
    # for each different sequence length
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = raw_model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f'sample {i}: {decoded}')


train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='val')
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


# optimize
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

train_steps, train_losses = [], []
val_steps, val_losses = [], []

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_interval == 0 or last_step:
        val_loss = evaluate_val_loss(model, val_loader, device)
        val_steps.append(step)
        val_losses.append(val_loss)

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % eval_interval == 0) or last_step:
        generate_samples(model, enc, device)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    # accumulate gradients over grad_accum_steps
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it come out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        
    # clip the gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    train_steps.append(step)
    train_losses.append(loss_accum.item())
    print(f'step {step}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}')

# plot train and validation loss on the same graph
plt.figure(figsize=(9, 5))
plt.plot(train_steps, train_losses, label='Train Loss', color='royalblue', linewidth=2)
plt.plot(val_steps, val_losses, label='Validation Loss', color='crimson', marker='o', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('train_val_loss.png', dpi=150)
plt.show()
