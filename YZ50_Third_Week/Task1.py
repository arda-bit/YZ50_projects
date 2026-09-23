import torch

words = open('names.txt', 'r').read().splitlines()

#b = {}
#for w in words:
#    chs = ['<S>'] + list(w) + ['<E>']
#    for ch1, ch2 in zip(chs, chs[1:]):
#        bigram = (ch1, ch2)
#        b[bigram] = b.get(bigram, 0) + 1
#        sorted(b.items() , key =lambda kv: -kv[1])

N = torch.zeros((27,  27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
   chs = ['.'] + list(w) + ['.']
   for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

import matplotlib.pyplot as plt

plt.figure(figsize=(18,18))
plt.imshow(N, cmap='Blues', interpolation='nearest')
for i in range(27):
    for j in range(27):
        chstr = itos[i] +itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=9)
        plt.text(j, i, N[i,j].item(), ha="center", va="top", color='gray', fontsize=8)
plt.axis('off')
plt.tight_layout()
plt.show()
