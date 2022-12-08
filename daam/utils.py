from functools import lru_cache
from pathlib import Path
import os
import sys
import random

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn.functional as F


__all__ = ['set_seed', 'compute_token_merge_indices', 'plot_mask_heat_map', 'cached_nlp', 'cache_dir']


def plot_mask_heat_map(im: PIL.Image.Image, heat_map: torch.Tensor, threshold: float = 0.4):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    return gen


def cache_dir() -> Path:
    # *nix
    if os.name == 'posix' and sys.platform != 'darwin':
        xdg = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        return Path(xdg, 'daam')
    elif sys.platform == 'darwin':
        # Mac OS
        return Path(os.path.expanduser('~'), 'Library/Caches/daam')
    else:
        # Windows
        local = os.environ.get('LOCALAPPDATA', None) \
                or os.path.expanduser('~\\AppData\\Local')
        return Path(local, 'daam')


def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None):
    merge_idxs = []

    if word_idx is None:
        prompt = prompt.lower()
        search_tokens = tokenizer.tokenize(word)
        punc_tokens = [p + '</w>' for p in string.punctuation]
        tokens = tokenizer.tokenize(prompt)
        # compute the tokens for each word
        word_tokens = [tokenizer.tokenize(word) for word in prompt.split()]

        # calculate the token position from the word position
        def calc_token_positions(end_idx, tokens_len):
            slice = word_tokens[:end_idx]
            first_pos = 0
            for word_token in slice:
                first_pos += len(word_token)

            # merge together all tokens in the word
            return [first_pos + i for i in range(0, tokens_len)]


        for idx, w_token in enumerate(word_tokens):
            # if the word contains more than one token
            if len(w_token) > len(search_tokens):
                # check to see if the extra tokens were from punctuation
                no_punc = [t for t in w_token if t not in punc_tokens]
                search_no_punc = [t for t in search_tokens if t not in punc_tokens]
                if no_punc and no_punc == search_no_punc:
                    merge_idxs += calc_token_positions(idx, len(search_tokens))
            elif w_token == search_tokens:
                merge_idxs += calc_token_positions(idx, len(search_tokens))
    else:
        merge_idxs.append(word_idx)

    # offset indices by one
    return [x + 1 for x in merge_idxs]


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type='en_core_web_md'):
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(type)
        except OSError:
            import os
            os.system(f'python -m spacy download {type}')
            nlp = spacy.load(type)

    return nlp(prompt)
