import torch

#create dataset
words = open('isimler', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

xs, ys = [], []

for w in words:
   chs = ['.'] + list(w) + ['.']
   for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

import torch.nn.functional as F

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((31,31), generator=g, requires_grad=True)

for i in range (500):
    xenc = F.one_hot(xs, num_classes = 31).float()

    logits = xenc @ W
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims=True)

    loss = -prob[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad

