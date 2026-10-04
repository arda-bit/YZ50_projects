import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B,T,C)

# single head perform self-attention

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # (B, T, 16)
q = query(x) # (B, T, 16)
v = value(x) # (B, T, 16)

wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) => (B, T, T)

print(f"wei.var(): {wei.var()}")

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

print(f"wei[0] (without sqrt(head_size) normalization): {wei[0]}")

wei = q @ k.transpose(-2, -1) * head_size**-0.5

print(f"wei.var(): {wei.var()}")

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

print(f"wei[0] (with sqrt(head_size) normalization): {wei[0]}")

