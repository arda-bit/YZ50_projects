import torch
import torch.nn.functional as F

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

    n_params = sum(p.nelement() for p in parameters) # number of parameters in total
    print(f'Number of parameters in this model: {n_params}')

    emb = C[X] # (228146, 3, 2)
    print(f'Embeddings Shape: {emb.shape}')

    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (228146, 100)
    logits = h @ W2 + b2 # (228146, 27)

    # manual loss calculation
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(X.shape[0]), Y].log().mean()
    print(f'manual cross-entropy loss calculation: {loss}')

    print(f'F.cross_entropy(logits, Y): {F.cross_entropy(logits, Y)}')
    

