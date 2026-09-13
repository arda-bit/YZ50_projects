import torch

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

    C = torch.randn((27, 2)) # d=2 dimensional embedding
    print(f'Embedding Matrix Shape: {C.shape}')

    emb = C[X]
    print(f'Embeddings: {emb.shape}')



