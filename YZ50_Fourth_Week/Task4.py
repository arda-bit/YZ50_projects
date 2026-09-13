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
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    # make the network larger
    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    C = torch.randn((27, 2), generator=g)
    W1 = torch.randn((6, 300), generator=g)
    b1 = torch.randn(300, generator=g)
    W2 = torch.randn((300, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True

    n_params = sum(p.nelement() for p in parameters) # number of parameters in total
    print(f'Number of parameters in this model: {n_params}')

    stepi = []
    lossi = []
    for i in range(60000):

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

        lr = 0.1
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        stepi.append(i)
        lossi.append(loss.item())
    
    plt.plot(stepi, lossi)
    plt.savefig('training_loss_curve.png')
    
    # learning rate decay
    for i in range(30000):

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

        lr = 0.01
        for p in parameters:
            p.data += -lr * p.grad

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
    training loss over training set: 2.268532
    validation loss over validation set: 2.275726
    """

    # plot the embeddings in 2D
    plt.figure(figsize=(8,8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
    plt.grid("minor")
    plt.savefig('embedding_scatter_plot.png')
    
    # Further work for hyperparameter tuning
    # 1. Increase the dimension of the hidden layer
    # 2. Increase the embedding dimension
    # 3. Increase the number of letter of context
    # 4. Increase the minibatch size - better convergence speed

    # sample names from the model
    g = torch.Generator().manual_seed(2147483647 + 10)
    block_size = 3

    for _ in range(20):

        out = []
        context = [0] * block_size # initialize with all ...
        while True:
            emb = C[torch.tensor([context])] # (1, block_size, d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        
        print(''.join(itos[i] for i in out))
    
    """
    carmahxami.
    havi.
    kimrix.
    thil.
    salani.
    emrahnel.
    ameraht.
    areei.
    nellaia.
    ceriia.
    asleigh.
    ham.
    poir.
    quint.
    salin.
    alian.
    qui.
    jero.
    dearixi.
    chae.
    """