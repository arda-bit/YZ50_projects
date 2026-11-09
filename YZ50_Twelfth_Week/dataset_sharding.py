"""
FineWeb-Edu dataset (for pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.

Run simply as:
> uv run python fineweb.py
Will save shards to the local directory "edu-fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken

from datasets import load_dataset
from tqdm import tqdm 

local_dir = 'edu_fineweb10B'
shard_size = int(1e7)   # 10M tokens per shard
max_shards = 5          # 5 shards total = 50M tokens (1 val + 4 train)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# load only the first parquet file instead of all 14 files
fw = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    data_files='sample/10BT/000_00000.parquet',
    split='train',
)

# init the tokenizer
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']  # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'token_dictionary too large to fit in memory'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shards_size tokens (last document might be shorter)
nprocs = max(1, os.cpu_count()//2)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
            progress_bar.update(len(tokens))

        else:
            # write the current shard and start a new one
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f'fineweb_{split}_{shard_index:06d}')

            # split the document into whatever fits in this shard; the remainder goes to final shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar.close()
            progress_bar = None
            if shard_index >= max_shards:
                token_count = 0
                break
            
            # populate the next shard with the leftover of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(DATA_CACHE_DIR, f'fineweb_{split}_{shard_index:06d}')
        write_datafile(filename, all_tokens_np[:token_count])

