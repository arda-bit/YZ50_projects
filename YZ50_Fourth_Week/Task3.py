import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def build_dataset(words):
    block_size = 3 # context length: how many chracters do we take to predict the next one?
    X, Y = [], []
    for w in words:
        
        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y



if __name__ == '__main__':
    # read in all the words
    words = open('names.txt', 'r').read().splitlines()

    print(f'Loaded {len(words)} words')

    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    print(itos)

    # build the dataset

    block_size = 3 # context length: how many chracters do we take to predict the next one?
    X, Y = [], []
    for w in words:
        
        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(f'X.shape: {X.shape}, X.dtype: {X.dtype}, \
        Y.shape: {Y.shape}, Y.dtype: {Y.dtype}')

    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    C = torch.randn((27, 2), generator=g)
    W1 = torch.randn((6, 100), generator=g)
    b1 = torch.randn(100, generator=g)
    W2 = torch.randn((100, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True

    n_params = sum(p.nelement() for p in parameters) # number of parameters in total
    print(f'Number of parameters in this model: {n_params}')

    ## ------------------------------------- ##
    ##        3.1 Overfit a minibatch        ##
    ## ------------------------------------- ##

    # minibatch construction
    ix = torch.randint(0, X.shape[0], (32,))

    for epoch in range(10000):

        emb = C[X[ix]] # (32, 3, 2)

        # forward pass
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Y[ix])
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        for p in parameters:
            p.data += -0.1 * p.grad

        if epoch % 100 == 0:
            print(f'epoch: {epoch} loss: {loss.item():4f}')

    print(f'Final loss to overfit minibatch: {loss.item():4f}')

    ## ------------------------------------- ##
    ##       3.2 Train with minibatches      ##
    ## ------------------------------------- ##

    for _ in range(1000):

        # minibatch construction
        ix = torch.randint(0, X.shape[0], (32,))

        # forward pass
        emb = C[X[ix]] # (32, 3, 2)
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Y[ix])
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        for p in parameters:
            p.data += -0.1 * p.grad

        if epoch % 100 == 0:
            emb = C[X] # (228146, 3, 2)
            h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Y)
            print(f'epoch: {epoch} loss: {loss.item():4f}')
    
        emb = C[X] # (228146, 3, 2)
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)
        print(f'Final loss over entire dataset: {loss.item():4f}')

    
    ## ------------------------------------- ##
    ##       3.3 Learning rate tuning        ##
    ## ------------------------------------- ##
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre

    lri = []
    lossi = []

    for i in range(1000):

        # minibatch construction
        ix = torch.randint(0, X.shape[0], (32,))

        # forward pass
        emb = C[X[ix]] # (32, 3, 2)
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Y[ix])
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        lr = lrs[i]
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        lri.append(lre[i])
        lossi.append(loss.item())

    plt.plot(lri, lossi)
    plt.savefig('learning_rate_tuning.png')


    ## ------------------------------------- ##
    ##       3.4 Train/Val/Test Split        ##
    ## ------------------------------------- ##
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    for epoch in range(3000):

        # minibatch construction
        ix = torch.randint(0, Xtr.shape[0], (32,))

        # forward pass
        emb = C[Xtr[ix]] # (32, 3, 2)
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32, 100)
        logits = h @ W2 + b2 # (32, 27)
        loss = F.cross_entropy(logits, Ytr[ix])
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        for p in parameters:
            p.data += -0.1 * p.grad

        if epoch % 100 == 0:
            emb = C[Xtr]
            h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Ytr)
            print(f'epoch: {epoch} training loss: {loss.item():4f}')
    
    # final training loss
    emb = C[Xtr] # (228146, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print(f'training loss over training set: {loss.item():4f}')

    # final validation loss
    emb = C[Xdev] # (228146, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(f'validation loss over validation set: {loss.item():4f}')

    """
    training loss over training set: 2.595423
    validation loss over validation set: 2.588420
    """
    