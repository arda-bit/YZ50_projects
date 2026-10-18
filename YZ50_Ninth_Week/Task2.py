
if __name__ == '__main__':

    with open('türkçe_metinler.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = text.encode("utf-8") # raw bytes
    tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(ids, pair, idx):
        # in the list of ints (ids), replace all consecutive occurence of pair with the new token idx
        newids = []
        i = 0
        while i < len(ids):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    vocab_size = 276 # the desired final vocabulary size
    num_merges = vocab_size - 256
    ids = list(tokens) # copy so we don't destroy the original list
    vocab = {idx: bytes([idx]) for idx in range(256)}


    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i

        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # Decode safely into strings
        p0_str = vocab[pair[0]].decode("utf-8", errors="replace")
        p1_str = vocab[pair[1]].decode("utf-8", errors="replace")
        new_str = vocab[idx].decode("utf-8", errors="replace")
        print(f"merging {pair} ({p0_str!r} + {p1_str!r}) -> new token {idx}: {new_str!r}")

        ids = merge(ids, pair, idx)
        merges[pair] = idx

    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

"""
merging (196, 177) ('�' + '�') -> new token 256: 'ı'
merging (110, 32) ('n' + ' ') -> new token 257: 'n '
merging (97, 114) ('a' + 'r') -> new token 258: 'ar'
merging (101, 32) ('e' + ' ') -> new token 259: 'e '
merging (101, 114) ('e' + 'r') -> new token 260: 'er'
merging (195, 188) ('�' + '�') -> new token 261: 'ü'
merging (97, 32) ('a' + ' ') -> new token 262: 'a '
merging (197, 159) ('�' + '�') -> new token 263: 'ş'
merging (105, 32) ('i' + ' ') -> new token 264: 'i '
merging (105, 110) ('i' + 'n') -> new token 265: 'in'
merging (108, 258) ('l' + 'ar') -> new token 266: 'lar'
merging (101, 116) ('e' + 't') -> new token 267: 'et'
merging (105, 108) ('i' + 'l') -> new token 268: 'il'
merging (105, 114) ('i' + 'r') -> new token 269: 'ir'
merging (46, 32) ('.' + ' ') -> new token 270: '. '
merging (97, 110) ('a' + 'n') -> new token 271: 'an'
merging (44, 32) (',' + ' ') -> new token 272: ', '
merging (107, 32) ('k' + ' ') -> new token 273: 'k '
merging (108, 260) ('l' + 'er') -> new token 274: 'ler'
merging (256, 32) ('ı' + ' ') -> new token 275: 'ı '
tokens length: 339183
ids length: 263224
compression ratio: 1.29X
"""