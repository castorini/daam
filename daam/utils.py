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


def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    prompt = prompt.lower()
    tokens = tokenizer.tokenize(prompt)
    word = word.lower()
    merge_idxs = []
    curr_idx = 0
    curr_token = ''

    if word_idx is None:
        try:
            word_idx = prompt.split().index(word, offset_idx)
        except:
            for punct in ('.', ',', '!', '?'):
                try:
                    word_idx = prompt.split().index(word + punct)
                    break
                except:
                    pass

    if word_idx is None:
        raise ValueError(f'Couldn\'t find "{word}" in "{prompt}"')

    for idx, token in enumerate(tokens):
        merge_idxs.append(idx)

        if '</w>' in token:
            curr_token += token[:-4]

            if idx >= word_idx and curr_token == word:
                break

            curr_token = ''
            curr_idx += 1
            merge_idxs.clear()
        else:
            curr_token += token
            merge_idxs.append(idx)

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
