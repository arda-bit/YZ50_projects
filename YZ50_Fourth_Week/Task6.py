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
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact) # (N, n_hidden)

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
    W2 = torch.randn((n_hidden, vocab_size),         generator=g)  * (5/3) / ((n_hidden)**0.5) # initialize to small values but not 0 (to preserve entropy/prevent symmetry)
    b2 = torch.randn((vocab_size,),                  generator=g)  * 0 # initialize to exactly 0

    # batch normalization
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden))
    bnstd_running = torch.ones((1, n_hidden))

    parameters = [C, W1, W2, b2, bngain, bnbias]
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
        hpreact = embcat @ W1 # hidden layer pre-activation

            # batch-normalization
        bnmeani = hpreact.mean(dim=0, keepdim=True)
        bnstdi = hpreact.std(dim=0, keepdim=True)

        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
        
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

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
    # w/o batch normalization
    # train 2.037996768951416
    # val 2.1054441928863525

    # with batch normalization
    # train 2.0711417198181152
    # val 2.11014461517334

    # sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)

    for _ in range(20):
        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            # forward pass the neural net
            emb = C[torch.tensor([context])] # (1, block_size,n_embd)
            hpreact = emb.view(1, -1) @ W1 # (1, block_size * n_embd) -> (1, hidden_size)
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact) # (N, n_hidden)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            #shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break
        
        print(''.join(itos[i] for i in out)) # decode and print the generated word

    """
    carmah.
    amelle.
    khyrmin.
    reety.
    salaysie.
    mahnel.
    delynn.
    jareei.
    ner.
    kia.
    chaiir.
    kaleigh.
    ham.
    joce.
    quinn.
    salin.
    alianni.
    watelo.
    dearyn.
    kai.
    """