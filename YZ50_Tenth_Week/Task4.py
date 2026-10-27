import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from Task2 import (
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
# device = "cpu"  # temporary override DELETE LATER

# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)


num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()

text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)


model = GPT(GPTConfig())
print(f'model initialized randomly!')
model.eval()
model.to(device)    # move the entire model to GPU


logits, loss = model(x, y)
print(f'logits.shape: {logits.shape}, loss: {loss}')

"""
logits.shape: torch.Size([4, 32, 50257]), loss: 10.92757797241211
-ln(1/50257) approx: 10.8249
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i}, loss: {loss.item()}') # .item() will port loss from GPU to CPU

"""
model initialized randomly!
logits.shape: torch.Size([4, 32, 50257]), loss: 10.92757797241211
step 0, loss: 10.92757797241211
step 1, loss: 6.5637383460998535
step 2, loss: 4.234703063964844
step 3, loss: 2.61612868309021
step 4, loss: 1.5009081363677979
step 5, loss: 0.8399207592010498
step 6, loss: 0.4715862572193146
step 7, loss: 0.28295770287513733
step 8, loss: 0.17427489161491394
step 9, loss: 0.11198063939809799
step 10, loss: 0.0735916793346405
step 11, loss: 0.055631570518016815
step 12, loss: 0.043171148747205734
step 13, loss: 0.03474025800824165
step 14, loss: 0.030323907732963562
step 15, loss: 0.027320001274347305
step 16, loss: 0.02399366721510887
step 17, loss: 0.020633410662412643
step 18, loss: 0.017856214195489883
step 19, loss: 0.01574232429265976
step 20, loss: 0.014126952737569809
step 21, loss: 0.012816664762794971
step 22, loss: 0.011663971468806267
step 23, loss: 0.010601235553622246
step 24, loss: 0.009628602303564548
step 25, loss: 0.008767278864979744
step 26, loss: 0.008026588708162308
step 27, loss: 0.007398046553134918
step 28, loss: 0.006863262504339218
step 29, loss: 0.006402835715562105
step 30, loss: 0.00600191205739975
step 31, loss: 0.005650036968290806
step 32, loss: 0.005339358001947403
step 33, loss: 0.005063202232122421
step 34, loss: 0.004815935622900724
step 35, loss: 0.004592967219650745
step 36, loss: 0.004390696529299021
step 37, loss: 0.0042064134031534195
step 38, loss: 0.004038019105792046
step 39, loss: 0.0038838882464915514
step 40, loss: 0.0037426231428980827
step 41, loss: 0.0036129499785602093
step 42, loss: 0.003493703668937087
step 43, loss: 0.003383811330422759
step 44, loss: 0.0032823500223457813
step 45, loss: 0.0031884037889540195
step 46, loss: 0.003101192880421877
step 47, loss: 0.003020043484866619
step 48, loss: 0.0029443935491144657
step 49, loss: 0.0028737266547977924
"""