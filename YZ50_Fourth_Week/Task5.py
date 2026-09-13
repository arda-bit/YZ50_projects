import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]

    emb = C[x] # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
    h = torch.tanh(embcat @ W1 + b1) # (N, n_hidden)
    logits = h @ W2 + b2 # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


if __name__ == '__main__':

    # read in all the words
    words = open('names.txt', 'r').read().splitlines()
    words[:8]

    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)


    block_size = 3 # context length: how many chracters do we take to predict the next one?
    
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])  # 80%
    Xdev, Ydev = build_dataset(words[n1:n2])  # 10%
    Xte, Yte = build_dataset(words[n2:])  # 10%

    # MLP revisited
    n_embd = 10  # the dimensionality of the character embedding vectors
    n_hidden = 200 # the number of neurons in the hidden layer of the MLP

    g = torch.Generator().manual_seed(2147483647)  # for reproducibility
    C = torch.randn((vocab_size, n_embd),            generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / ((n_embd * block_size)**0.5) # Kaiming initialization
    b1 = torch.randn((n_hidden,),                    generator=g)  * 0.01
    W2 = torch.randn((n_hidden, vocab_size),         generator=g)  * (5/3) / ((n_hidden)**0.5) # initialize to small values but not 0 (to preserve entropy/prevent symmetry)
    b2 = torch.randn((vocab_size,),                  generator=g)  * 0 # initialize to exactly 0

    parameters = [C, W1, b1, W2, b2]
    print(sum(p.nelement() for p in parameters))  # number of parameters in total
    for p in parameters:
        p.requires_grad = True
    
    max_steps = 200000
    batch_size = 32
    lossi = []

    for i in range(max_steps):

        # minibatch construction
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]  # batch X, Y

        # forward pass
        emb = C[Xb] # embed the characters into vectors
        embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
        hpreact = embcat @ W1 + b1 # hidden layer pre-activation
        h = torch.tanh(hpreact) # hidden layer
        logits = h @ W2 + b2 # output layer
        loss = F.cross_entropy(logits, Yb) # loss function

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
    
    split_loss('train')
    split_loss('val')
    # train 2.037996768951416
    # val 2.1054441928863525

    plt.hist(h.view(-1).tolist(), 50)
    plt.savefig('saturated_tanh_historgram.png')

    plt.hist(hpreact.view(-1).tolist(), 50)
    plt.savefig('preactivation_historgram.png')
