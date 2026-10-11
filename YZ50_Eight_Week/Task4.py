import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1337)

# read it in to inspect it
with open('türkçe_metinler.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)


n = int(0.9 * len(data))  # first 90% will be train, rest validation
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_intervals = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 128
n_head = 6
n_layer = 4
dropout = 0.2

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Single Self-Attention Module
class SelfAttentionHead(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, ):
    B,T,C = x.shape
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)

    # compute attention scores
    wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) => (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) => (B,T,C)
    return out 

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigrame Baseline
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    token_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = token_emb + pos_emb # (B,T,C)
    x = self.blocks(x)
    logits = self.lm_head(x)  # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, _ = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel()
model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f"step: {iter}: train loss: {losses["train"]:.4f}, val loss: {losses["val"]:4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device) # 0 token id is the '\n' character
print(f"Generated text: {decode(model.generate(idx, max_new_tokens=500)[0].tolist())}")

"""
step: 0: train loss: 5.0349, val loss: 5.025891
step: 500: train loss: 2.1120, val loss: 2.151912
step: 1000: train loss: 1.7873, val loss: 1.901156
step: 1500: train loss: 1.5837, val loss: 1.760581
step: 2000: train loss: 1.4575, val loss: 1.678574
step: 2500: train loss: 1.3685, val loss: 1.633075
step: 3000: train loss: 1.3033, val loss: 1.590308
step: 3500: train loss: 1.2551, val loss: 1.570851
step: 4000: train loss: 1.2099, val loss: 1.555380
step: 4500: train loss: 1.1777, val loss: 1.543537
Generated text: 
Abbasî halifeleri, vazifelerin bilhassa sukunun sürdürmelerini aldılar. 
(Bullarile Bağdadın enaların bilgisi kuvvetleri az berlerden bi şerasil de olduktan itaatler ve başlıyan müteskiyetinden 
muharebesi Felh sasrayından istikasına memleğen kalpanları müzakebe yürüddüler. Araplarile yakın bir itibarile muahede Valisi kumanlanan, 
Eşruseler mevhez ve şehrine dahilere kolay etmemişti. Bu histiyatın biribir altında bulunduğu Barada başka yetler Türkiyeyan dolguer har, 
yılın aölde ortarlar; türk birç
"""