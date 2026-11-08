import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
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

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)   # inputs
        y = (buf[1:]).view(B, T)    # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
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

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it+1) / warmup_iters
    # 2) if it > max_stes, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# plot the learning rate schedule
steps = list(range(max_steps))
lrs = [get_lr(it) for it in steps]

plt.figure(figsize=(8, 5))
plt.plot(steps, lrs, label='Learning Rate', color='royalblue', linewidth=2)
plt.axvline(x=warmup_iters - 1, color='crimson', linestyle='--', alpha=0.7, label=f'Warmup End (step {warmup_iters - 1})')
plt.title('Cosine Learning Rate Schedule with Linear Warmup')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('lr_schedule.png', dpi=150)
plt.show()


train_loader = DataLoaderLite(B=B, T=T)
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


# optimize
# optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    
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
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

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
    print(f'step {step}, loss: {loss_accum.item()}, lr: {lr:.4f}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec}')

"""
using device: cuda
total desired batch size: 524288
=> calculated gradient accumulation steps: 64

loaded 338025 tokens
1 epoch = 41 batches
num decay parameter tensors: 50, with 124354560 parameters
num non-decay parameter tensors: 98, with 121344 parameters
using fused AdamW: True
step 0, loss: 10.939042091369629, lr: 0.0001, norm: 27.0273, dt: 18176.43ms, tok/sec: 28844.393351987244
step 1, loss: 9.647464752197266, lr: 0.0001, norm: 9.5057, dt: 16635.77ms, tok/sec: 31515.695073697505
step 2, loss: 9.226698875427246, lr: 0.0002, norm: 5.7455, dt: 16938.10ms, tok/sec: 30953.174641721496
step 3, loss: 9.812556266784668, lr: 0.0002, norm: 8.3081, dt: 17097.00ms, tok/sec: 30665.496651848465
step 4, loss: 9.20470905303955, lr: 0.0003, norm: 4.2689, dt: 17026.38ms, tok/sec: 30792.696093866685
step 5, loss: 8.671659469604492, lr: 0.0004, norm: 3.6468, dt: 16852.50ms, tok/sec: 31110.39736048542
step 6, loss: 8.310739517211914, lr: 0.0004, norm: 1.9520, dt: 18194.48ms, tok/sec: 28815.765612866
step 7, loss: 8.057013511657715, lr: 0.0005, norm: 2.7684, dt: 18054.69ms, tok/sec: 29038.87842858834
step 8, loss: 7.712160587310791, lr: 0.0005, norm: 1.8915, dt: 18174.76ms, tok/sec: 28847.042040662513
step 9, loss: 7.355413913726807, lr: 0.0006, norm: 1.7914, dt: 18390.14ms, tok/sec: 28509.189423889697
step 10, loss: 7.020171165466309, lr: 0.0006, norm: 1.8460, dt: 18427.68ms, tok/sec: 28451.115962869964
step 11, loss: 6.762916088104248, lr: 0.0006, norm: 1.5049, dt: 18576.76ms, tok/sec: 28222.791201091793
step 12, loss: 6.50872802734375, lr: 0.0006, norm: 1.1007, dt: 18394.56ms, tok/sec: 28502.337822207486
step 13, loss: 6.394480228424072, lr: 0.0006, norm: 1.0550, dt: 18549.04ms, tok/sec: 28264.962695425278
step 14, loss: 6.35346794128418, lr: 0.0006, norm: 2.6706, dt: 18571.47ms, tok/sec: 28230.83038475121
step 15, loss: 6.237180233001709, lr: 0.0006, norm: 0.8995, dt: 18452.16ms, tok/sec: 28413.37374741341
step 16, loss: 6.232109546661377, lr: 0.0006, norm: 0.7949, dt: 18523.79ms, tok/sec: 28303.501135049777
step 17, loss: 6.193635940551758, lr: 0.0006, norm: 1.1215, dt: 18574.91ms, tok/sec: 28225.59758345232
step 18, loss: 6.17728853225708, lr: 0.0005, norm: 1.0806, dt: 18490.02ms, tok/sec: 28355.195988319898
step 19, loss: 6.144557952880859, lr: 0.0005, norm: 1.1747, dt: 19149.02ms, tok/sec: 27379.370100014574
step 20, loss: 6.16644811630249, lr: 0.0005, norm: 1.1384, dt: 18588.01ms, tok/sec: 28205.704859036072
step 21, loss: 6.087320327758789, lr: 0.0005, norm: 0.6200, dt: 21600.83ms, tok/sec: 24271.66546965612
step 22, loss: 6.052770614624023, lr: 0.0005, norm: 0.5327, dt: 20563.83ms, tok/sec: 25495.63802596516
step 23, loss: 6.038298606872559, lr: 0.0005, norm: 0.8369, dt: 18760.24ms, tok/sec: 27946.769218643156
step 24, loss: 6.00568962097168, lr: 0.0005, norm: 0.3975, dt: 18339.13ms, tok/sec: 28588.49454816414
step 25, loss: 5.986323356628418, lr: 0.0004, norm: 0.4809, dt: 18509.29ms, tok/sec: 28325.66018187226
step 26, loss: 5.971500396728516, lr: 0.0004, norm: 0.3984, dt: 18413.96ms, tok/sec: 28472.317522723784
step 27, loss: 5.981678009033203, lr: 0.0004, norm: 0.4916, dt: 19023.73ms, tok/sec: 27559.68115618395
step 28, loss: 5.942702293395996, lr: 0.0004, norm: 0.3411, dt: 18155.31ms, tok/sec: 28877.94045536708
step 29, loss: 5.966236591339111, lr: 0.0004, norm: 0.3658, dt: 18417.51ms, tok/sec: 28466.821255172632
step 30, loss: 5.939932823181152, lr: 0.0003, norm: 0.3710, dt: 18625.40ms, tok/sec: 28149.093116624903
step 31, loss: 5.9343743324279785, lr: 0.0003, norm: 0.3146, dt: 18489.64ms, tok/sec: 28355.772224500863
step 32, loss: 5.9286603927612305, lr: 0.0003, norm: 0.3983, dt: 18454.59ms, tok/sec: 28409.628082006708
step 33, loss: 5.916708946228027, lr: 0.0003, norm: 0.2931, dt: 18601.32ms, tok/sec: 28185.52043419649
step 34, loss: 5.909920692443848, lr: 0.0002, norm: 0.2512, dt: 18642.50ms, tok/sec: 28123.271520025264
step 35, loss: 5.89404821395874, lr: 0.0002, norm: 0.2782, dt: 18490.89ms, tok/sec: 28353.85567165886
step 36, loss: 5.906646728515625, lr: 0.0002, norm: 0.1843, dt: 18543.98ms, tok/sec: 28272.672599330916
step 37, loss: 5.870262622833252, lr: 0.0002, norm: 0.2902, dt: 18860.79ms, tok/sec: 27797.77831421543
step 38, loss: 5.883539199829102, lr: 0.0002, norm: 0.2996, dt: 18420.95ms, tok/sec: 28461.503926358197
step 39, loss: 5.861596584320068, lr: 0.0002, norm: 0.2435, dt: 18486.70ms, tok/sec: 28360.282739768823
step 40, loss: 5.8524603843688965, lr: 0.0001, norm: 0.2047, dt: 18583.01ms, tok/sec: 28213.296676433103
step 41, loss: 5.849183082580566, lr: 0.0001, norm: 0.2095, dt: 18486.19ms, tok/sec: 28361.064381579214
step 42, loss: 5.83150577545166, lr: 0.0001, norm: 0.2113, dt: 20000.87ms, tok/sec: 26213.263486775144
step 43, loss: 5.838803291320801, lr: 0.0001, norm: 0.2664, dt: 18570.20ms, tok/sec: 28232.757527073343
step 44, loss: 5.812167644500732, lr: 0.0001, norm: 0.2326, dt: 18643.96ms, tok/sec: 28121.05721821413
step 45, loss: 5.835803508758545, lr: 0.0001, norm: 0.1848, dt: 18413.88ms, tok/sec: 28472.435122986284
step 46, loss: 5.811927318572998, lr: 0.0001, norm: 0.1993, dt: 18886.73ms, tok/sec: 27759.600908667282
step 47, loss: 5.812937259674072, lr: 0.0001, norm: 0.1949, dt: 18740.69ms, tok/sec: 27975.91338199225
step 48, loss: 5.8023858070373535, lr: 0.0001, norm: 0.1855, dt: 18581.36ms, tok/sec: 28215.81045057658
step 49, loss: 5.802640914916992, lr: 0.0001, norm: 0.1933, dt: 18588.45ms, tok/sec: 28205.043179643126
"""