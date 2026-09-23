import torch
import matplotlib.pyplot as plt
import math

words = open('names.txt').read().splitlines()

M = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        M[ix1, ix2] += 1

prob = M[0].float()
prob = prob / prob.sum()

g = torch.Generator().manual_seed(2147483647)
idx = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
itos[idx]

#27, 27
#27,  1

#27, 27
# 1  27

P = M.float()
P /= P.sum(1, keepdim=True)

plt.figure(figsize=(18,18))
plt.imshow(P, cmap='Blues', interpolation='nearest')
for i in range(27):
    for j in range(27):
        chstr = itos[i] +itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=9)
        plt.text(j, i, round(P[i,j].item(), 3), ha="center", va="top", color='gray', fontsize=6)
plt.axis('off')
plt.tight_layout()
plt.show()


for i in range(10):

    out = []
    ix = 0
    while True:
        
        p=P[ix]
        #p = M[ix].float()
        #p = p / p.sum()
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
