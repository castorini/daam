from functools import lru_cache
from pathlib import Path
import random

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn.functional as F


__all__ = ['expand_image', 'set_seed', 'compute_token_merge_indices', 'plot_overlay_heat_map', 'plot_mask_heat_map',
           'cached_nlp']


def expand_image(im: torch.Tensor, out: int = 512, absolute: bool = False, threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(out, out), mode='bicubic')

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    im = im.cpu().detach()

    return im.squeeze()


def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool) -> None
    plt.clf()
    plt.rcParams.update({'font.size': 24})

    with torch.cuda.amp.autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
        else:
            heat_map = heat_map.clamp_(min=0, max=1)
            plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt.imshow(im)

        if word is not None:
            plt.title(word)

        if out_file is not None:
            plt.savefig(out_file)


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


def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None):
    prompt = prompt.lower()
    tokens = tokenizer.tokenize(prompt)
    word = word.lower()
    merge_idxs = []
    curr_idx = 0
    curr_token = ''

    if word_idx is None:
        try:
            word_idx = prompt.split().index(word)
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

            if curr_idx == word_idx and curr_token == word:
                break

            curr_token = ''
            curr_idx += 1
            merge_idxs.clear()
        else:
            curr_token += token
            merge_idxs.append(idx)

    return [x + 1 for x in merge_idxs]  # Offset by 1.


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type='en_core_web_md'):
    global nlp

    if nlp is None:
        nlp = spacy.load(type)

    return nlp(prompt)
