from collections import defaultdict
from pathlib import Path
import argparse
import cv2
import mmcv

import torch
import numpy as np

from dib.run.analyze import expand_m
from diffusers import StableDiffusionPipeline

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


def compute_iou(pred, target) -> float:
    """
    Compute the intersection over union of masks
    """
    pred = pred.view(-1)
    target = target.view(-1)

    if pred.sum().item() == 0:
        return 0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection / union).item()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--segment', type=str, default='80', choices=['infty', '80'])
    parser.add_argument('--max-threshold', '-t', type=float, default=0.3)
    parser.add_argument('--aux-model', type=str, default='mask2former', choices=['mask2former', 'queryinst', 'maskrcnn', 'random'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iou-threshold', '-iout', type=float, default=0.5)
    args = parser.parse_args()
    #
    merge_dict = {
        'people': 'person',
        'guy': 'person',
        'kid': 'person',
        'kids': 'person',
        'woman': 'person',
        'men': 'person',
        'women': 'person',
        'girl': 'person',
        'girls': 'person',
        'boy': 'person',
        'boys': 'person',
        'man': 'person',
        'oven': 'appliances',
        'fire hydrant': 'hydrant',
        'stop sign': 'sign',
        'sports ball': 'ball',
        'baseball bat': 'bat',
        'baseball glove': 'glove',
        'tennis racket': 'racket',
        'wine glass': 'glass',
        'donut': 'doughnut',
        'potted plant': 'plant',
        'dining table': 'table',
        'cell phone': 'phone',
        'refrigerator': 'appliances',
        'teddy bear': 'bear',
        'hair drier': 'drier',
        'couch': 'chair'
    }

    merge_dict = {}

    global CLASSES
    CLASSES = [merge_dict.get(c, c) for c in CLASSES]

    input_folder = Path(args.input_folder)
    paths = list(input_folder.rglob('**/*.gt.png'))

    daam_dict = defaultdict(list)
    aux_dict = defaultdict(list)
    aps_daam = []
    aps_aux = []
    all_ious_daam = []
    all_ious_aux = []
    all_ioas_daam = []
    all_ioas_aux = []

    for path in paths:
        path = path.absolute()
        word = path.name.split('.')[0]

        for idx in range(30):
            attr_file = path.parent / f'{word}.{idx}.attrmap.2-4-8.pt'
            if attr_file.exists():
                break

        try:
            m = torch.load(str(attr_file))
        except FileNotFoundError:
            continue

        m = expand_m(m)
        mask = torch.ones_like(m)
        mask[m < args.max_threshold] = 0

        gt_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.float32) / 255
        gt_map[gt_map < 1.0] = 0
        gt_map = torch.from_numpy(gt_map)

        if args.segment == '80':
            word = merge_dict.get(word, word)

            if word not in CLASSES:
                continue

        iou = compute_iou(mask, gt_map)
        daam_dict[word].append(iou)
        aps_daam.append(iou)
        all_ious_daam.append(iou)
        all_ioas_daam.append(compute_ioa(mask, gt_map))

        if args.aux_model != 'random':
            aux_model_file = path.parent / f'_masks.pred.{args.aux_model}.pt'

            if aux_model_file.exists():
                if args.segment != '80' and word not in CLASSES:
                    aux_dict[word].append(0)
                    all_ioas_aux.append(0)
                    aps_aux.append(0)
                    continue

                bbox_result, masks = torch.load(aux_model_file)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]

                labels = np.concatenate(labels)
                bboxes = np.vstack(bbox_result)
                num_masks = 0

                if bboxes[:, :4].sum() == 0:
                    x_any = np.any(masks, axis=1)
                    y_any = np.any(masks, axis=2)
                    num_masks = len(bboxes)

                    for idx in range(num_masks):
                        x = np.where(x_any[idx, :])[0]
                        y = np.where(y_any[idx, :])[0]
                        bboxes[idx, :4] = np.array(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1],
                            dtype=np.float32)

                if masks is not None:
                    scores = bboxes[:, -1]
                    inds = scores > args.max_threshold
                    labels = labels[inds]
                    # print(labels)
                    masks = masks[inds, ...]
                    # print(masks.shape)
                    total_mask = 0

                    for lbl, m in zip(labels, masks):
                        if lbl == CLASSES.index(word):
                            # print(lbl)
                            total_mask += m

                    aux_pred = torch.tensor(total_mask).float().clamp(0, 1)
                    iou = compute_iou(aux_pred, gt_map)
                    aux_dict[word].append(iou)
                    aps_aux.append(iou)
                    all_ious_aux.append(iou)
                    all_ioas_aux.append(compute_ioa(aux_pred, gt_map))
                    # print(word, aux_dict[word])
                else:
                    aux_dict[word].append(0)
                    all_ioas_aux.append(0)
                    aps_aux.append(0)
            else:
                aux_dict[word].append(0)
                all_ioas_aux.append(0)
                aps_aux.append(0)
        else:
            a = np.arange(gt_map.view(-1).size(0))
            a_idxs = np.random.choice(a, size=gt_map.view(-1).size(0) // 2, replace=False)
            aux_pred = gt_map.clone()
            aux_pred.fill_(1.0)
            aux_pred.view(-1)[a_idxs] = 0

            iou = compute_iou(aux_pred, gt_map)
            aux_dict[word].append(iou)
            aps_aux.append(iou)
            all_ious_aux.append(iou)
            all_ioas_aux.append(compute_ioa(aux_pred, gt_map))

    daam_all_iou = np.array(all_ious_daam)
    aux_all_iou = np.array(all_ious_aux)

    print('DAAM: ', compute_map(daam_dict.values()))
    print(f'{args.aux_model}: '.capitalize(), compute_map(aux_dict.values()))
    print(f'DAAM P{args.iou_threshold}: ', np.sum((daam_all_iou > args.iou_threshold).astype(np.float32)) / len(daam_all_iou))
    print(f'{args.aux_model} P{args.iou_threshold}: ', np.sum((aux_all_iou > args.iou_threshold).astype(np.float32)) / len(aux_all_iou))
    print(f'DAAM s_h: ', np.mean(all_ioas_daam))
    print(f'{args.aux_model} s_h: ', np.mean(all_ioas_aux))
    print('DAAM AP: ', compute_ap(aps_daam))
    print('AUX AP: ', compute_ap(aps_aux))


if __name__ == '__main__':
    main()
