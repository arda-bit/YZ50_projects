import torch
import torch.nn.functional as F


torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)

# Version 1: Loop
# we want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
  for t in range(T):
    xprev = x[b, :t+1]  # (t, C)
    xbow[b,t] = torch.mean(xprev, 0)


# Version 2: Vectorized
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) => (B, T, C)


# Version 3: Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))

wei = F.softmax(wei, dim=-1) # Using softmax
xbow3 = wei @ x

print(f'torch.allclose(xbow, xbow2, atol=1e-7): {torch.allclose(xbow, xbow2, atol=1e-7)}')
print(f'torch.allclose(xbow, xbow3, atol=1e-7): {torch.allclose(xbow, xbow3, atol=1e-7)}')