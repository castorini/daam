import string
from functools import lru_cache
from pathlib import Path
import os
import sys
import random
from typing import TypeVar

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn.functional as F


__all__ = ['set_seed', 'compute_token_merge_indices', 'plot_mask_heat_map', 'cached_nlp', 'cache_dir', 'auto_device', 'auto_autocast']


T = TypeVar('T')


def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)


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

    gen = torch.Generator(device=auto_device())
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


def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    prompt = prompt.lower()
    tokens = tokenizer.tokenize(prompt)
    word = word.lower()
    merge_idxs = []
    curr_idx = 0
    curr_token = ''

    if word_idx is None:
        prompt = prompt.lower()
        search_tokens = tokenizer.tokenize(word)
        punc_tokens = [p + '</w>' for p in string.punctuation]
        # compute the tokens for each word
        word_tokens = [tokenizer.tokenize(word) for word in prompt.split()]

        # calculate the token position from the word position
        def calc_token_positions(end_idx, token_len):
            slice = word_tokens[:end_idx]
            first_pos = 0
            for word_token in slice:
                first_pos += len(word_token)

            # merge together all tokens in the word
            return [first_pos + i + offset_idx for i in range(0, token_len)]

        for idx, w_token in enumerate(word_tokens):
            # if the word contains more than one token
            if len(w_token) > len(search_tokens):
                # check to see if the extra tokens were from punctuation
                no_punc = [t for t in w_token if t not in punc_tokens]
                search_no_punc = [t for t in search_tokens if t not in punc_tokens]
                if no_punc and no_punc == search_no_punc:
                    merge_idxs += calc_token_positions(idx, len(search_tokens))
                    word_idx = idx
            elif w_token == search_tokens:
                merge_idxs += calc_token_positions(idx, len(search_tokens))
                word_idx = idx
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


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
