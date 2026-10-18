
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
        # print(f"merging {pair} ({p0_str!r} + {p1_str!r}) -> new token {idx}: {new_str!r}")

        ids = merge(ids, pair, idx)
        merges[pair] = idx
    
    def decode(ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace') # standard practice is to replace when byte isn't valid utf-8
        return text
    
    def encode(text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float('inf')))

            if pair not in merges:
                break # nothing else can be merged
            idx = merges[pair]
            tokens = merge(tokens, pair, idx)
        
        return tokens
    

    turkish_pangram = "Pijamalı hasta, yağız şoföre çabucak güvendi 😂."
    print(f"encode: {encode(turkish_pangram)}")
    print(f"decode(encode): {decode(encode(turkish_pangram))}")
    print(f"text == decode(encode): {decode(encode(turkish_pangram)) == turkish_pangram}")

"""
encode: [80, 105, 106, 97, 109, 97, 108, 275, 104, 97, 115, 116, 97, 272, 121, 97, 196, 159, 256, 122, 32, 263, 111, 102, 195, 182, 114, 259, 195, 167, 97, 98, 117, 99, 97, 273, 103, 261, 118, 101, 110, 100, 264, 240, 159, 152, 130, 46]
decode(encode): Pijamalı hasta, yağız şoföre çabucak güvendi 😂.
text == decode(encode): True
"""