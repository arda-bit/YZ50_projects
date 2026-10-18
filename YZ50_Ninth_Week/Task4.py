import regex as re

GPT_2_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_stats(ids):
    counts = {}
    for chunk in ids:
        for pair in zip(chunk, chunk[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_chunk(chunk, pair, idx):
    new_chunk = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i+1] == pair[1]:
            new_chunk.append(idx)
            i += 2
        else:
            new_chunk.append(chunk[i])
            i += 1
    return new_chunk

def merge(ids, pair, idx):
    return [merge_chunk(chunk, pair, idx) for chunk in ids]

def train_bpe(corpus_path='türkçe_metinler.txt', vocab_size=276, verbose=True):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_chunks = re.findall(GPT_2_PATTERN, text)
    ids = [list(ch.encode("utf-8")) for ch in text_chunks]

    num_merges = vocab_size - 256
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i

        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
            p0_str = vocab[pair[0]].decode("utf-8", errors="replace")
            p1_str = vocab[pair[1]].decode("utf-8", errors="replace")
            new_str = vocab[idx].decode("utf-8", errors="replace")
            print(f"merging {pair} ({p0_str!r} + {p1_str!r}) -> new token {idx}: {new_str!r}")

        ids = merge(ids, pair, idx)
        merges[pair] = idx

    def decode(ids_list):
        tokens = b"".join(vocab[idx] for idx in ids_list)
        return tokens.decode("utf-8", errors='replace')

    def encode(raw_text):
        chunks = re.findall(GPT_2_PATTERN, raw_text)
        all_ids = []
        for chunk in chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = {}
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
                pair = min(stats, key=lambda p: merges.get(p, float('inf')))

                if pair not in merges:
                    break
                idx = merges[pair]
                chunk_ids = merge_chunk(chunk_ids, pair, idx)
            all_ids.extend(chunk_ids)
        return all_ids

    return encode, decode, vocab, merges


if __name__ == '__main__':

    print(re.findall(GPT_2_PATTERN, "Hello've world123, how are you?"))

    print(re.findall(GPT_2_PATTERN, """Sonra "Ankara'dan" ve "İstanbul'da" gibi kesme işaretli kelimeleri bu desenle böl."""))

    encode, decode, vocab, merges = train_bpe(verbose=True)

    turkish_pangram = "Pijamalı hasta, yağız şoföre çabucak güvendi 😂."
    encoded = encode(turkish_pangram)
    decoded = decode(encoded)
    print(f"encode: {encoded}")
    print(f"decode(encode): {decoded}")
    print(f"text == decode(encode): {decoded == turkish_pangram}")

"""
['Hello', "'ve", ' world', '123', ',', ' how', ' are', ' you', '?']
['Sonra', ' "', 'Ankara', "'d", 'an', '"', ' ve', ' "', 'İstanbul', "'d", 'a', '"', ' gibi', ' kesme', ' işaretli', ' kelimeleri', ' bu', ' desenle', ' böl', '.']

Task 2'de harfler genellikle sonlarındaki boşluklarla kaynaşıp kirlenirken ('n ', 'e '), GPT-2 regex deseni kelimeleri izole ederek karakterlerin yalnızca kendi sözcük sınırları içinde birleşmesini sağlamıştır.
Bu izolasyon sayesinde birleştirme kotası boşluklara heba edilmemiş; Türkçe'deki bulunma durumu ('de', 'da'), tamlayan/iyelik eki ('ın', 'in') ve çoğul ekleri ('lar', 'ler') saf halleriyle ortaya çıkmıştır.
Sonuç olarak model, Türkçenin sondan eklemeli morfolojik yapısını ve kök-ek ilişkilerini çok daha tutarlı, temiz ve anlamlı alt-kelimeler (subwords) halinde yakalayabilmiştir.
"""