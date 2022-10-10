import networkx as nx
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
from pathlib import Path
import argparse
import cv2
import mmcv
import re

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from nltk.corpus import wordnet as wn

from dib.run.analyze import expand_m, nlp_cache

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def compute_ioa(a, b) -> float:
    """
    Compute the intersection over area of masks
    """
    a = a.view(-1)
    b = b.view(-1)

    intersection = (a * b).sum()
    area = a.sum()

    if area.item() == 0:
        return 0

    return (intersection / area).item()


def compute_iou(pred, target) -> float:
    """
    Compute the intersection over union of masks
    """
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection / union).item()


def compute_corr(a, b):
    a = a.view(-1)
    b = b.view(-1)

    return spearmanr(a.cpu().numpy(), b.cpu().numpy())[0]


def compute_ap(ious) -> float:
    if ious is None or len(ious) == 0:
        return None

    ious = sorted(ious, reverse=True)
    ap = 0
    last_iou = 1

    for i, iou in enumerate(ious):
        width = last_iou - iou
        ap += width * i / len(ious)
        last_iou = iou

    ap += last_iou

    return ap


def compute_map(ious_lst) -> float:
    aps = [compute_ap(ious) for ious in ious_lst]
    aps = [ap for ap in aps if ap is not None]

    if len(aps) == 0:
        return None

    return sum(aps) / len(aps)


def compute_intensity(mask):
    return (mask.sum().float() / mask.numel()).item()


def compute_graph(doc):
    import networkx as nx
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    # Get the length and path
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--max-threshold', '-t', type=float, default=0.4)
    parser.add_argument('--aux-model', type=str, default='mask2former', choices=['mask2former', 'queryinst'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    adjective_ioas = []
    random_ioas = []
    intensities = defaultdict(list)
    dep_ioas = defaultdict(list)
    syntax_tree_dists = []

    for path in tqdm(list(input_folder.iterdir())):
        if not path.is_dir():
            continue

        patt = re.compile(r'(.+?)\.(\d+)\.attrmap.2-4-8.pt')
        prompt = (path / 'prompt.txt').read_text().strip().lower()
        doc = nlp_cache(prompt, 'en_core_web_trf')
        # graph = compute_graph(doc)

        for file in path.iterdir():
            match = patt.match(file.name)
            if match is None:
                continue

            target_word = match.group(1)
            word_idx = int(match.group(2))

            try:
                nlp_word = doc[word_idx]
            except IndexError:
                continue

            if nlp_word.text.lower() != target_word.lower():
                continue

            m = torch.load(str(file))
            m = expand_m(m, abs_=False)
            mask = torch.ones_like(m)
            mask[m < args.max_threshold] = 0
            intensities[nlp_word.pos_].append(compute_intensity(mask))
            # intensities[nlp_word.pos_].append(m.sum().item())
            #
            # for _ in range(3):
            #     rand_word = doc[np.random.randint(len(doc))]
            #
            #     if rand_word.text.lower() == nlp_word.text.lower():
            #         continue
            #
            #     rand_widx = prompt[:rand_word.idx].count(' ')
            #     rand_file = path / f'{rand_word.text}.{rand_widx}.attrmap.2-4-8.pt'
            #
            #     if rand_file.exists() and nlp_word.pos_ == 'NOUN' and nlp_word.pos_ == rand_word.pos_:
            #         val = nlp_word.similarity(rand_word)
            #
            #         child_m = torch.load(str(rand_file))
            #         child_m = expand_m(child_m, abs_=True)
            #         child_mask = torch.ones_like(child_m)
            #         child_mask[child_m < args.max_threshold] = 0
            #
            #         iou = compute_iou(child_mask, mask)
            #         syntax_tree_dists.append((val, iou))

            # if nlp_word.pos_ == 'NUM':
                # print('parent', nlp_word)
                # print(nlp_word, path, intensities[nlp_word.pos_][-1])

                # plt.imshow(mask.cpu().detach().numpy())
                # plt.show()

            for child in nlp_word.children:
                words = prompt.split(' ')
                child_widx = prompt[:child.idx].count(' ')

                if words[child_widx] != child.text.lower():
                    continue

                child_file = path / f'{child.text.lower()}.{child_widx}.attrmap.2-4-8.pt'

                if not child_file.exists():
                    continue

                child_m = torch.load(str(child_file))
                child_m = expand_m(child_m, abs_=False)
                child_mask = torch.ones_like(child_m)
                child_mask[child_m < args.max_threshold] = 0

                ioa = compute_iou(child_mask, mask)

                if np.isnan(ioa):
                    continue

                dep_ioas[(nlp_word.pos_, child.dep_, child.pos_)].append(compute_ioa(mask, child_mask))
                # plt.clf()

    print('Adjective IOAs:', np.mean(adjective_ioas))
    print('Random IOAs:', np.mean(random_ioas))

    for pos, intensities in intensities.items():
        print(f'{pos} intensity:', f'{np.mean(intensities):.4f}', len(intensities))

    for rel, ioa in dep_ioas.items():
        if len(ioa) < 10:
            continue
        print(f'{rel} ioa:', np.mean(ioa), len(ioa))

    print('Syntax tree corr. dist:', spearmanr(*zip(*syntax_tree_dists)))


if __name__ == '__main__':
    main()
