from collections import defaultdict
from pathlib import Path
import argparse

from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from transformers import AutoTokenizer
import numpy as np
import scipy.stats

from daam import GenerationExperiment, HeatMap, plot_overlay_heat_map, expand_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--pred-prefix', '-p', type=str, default='daam')
    parser.add_argument('--model', type=str, default='/home/ralph/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/c47dae76a67f9470c56695af441b28f8a16f7056/tokenizer')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)

    num_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    Xs = defaultdict(list)
    ys = defaultdict(list)
    feats = defaultdict(list)
    y = []

    num_preds = []
    num_trues = []
    num_prompts = []

    for path in input_folder.iterdir():
        b = 64
        if not path.is_dir() or not GenerationExperiment.has_annotations(path):
            continue

        exp = GenerationExperiment.load(str(path), args.pred_prefix)
        num_objects = exp.annotations['num_objects']

        if num_objects < 0 or num_objects > 9:
            continue

        # Run test on exp.prompt.split()[-1] as well...
        heat_map = HeatMap(tokenizer, exp.prompt, exp.global_heat_map)
        heat_map2 = expand_image(heat_map.compute_word_heat_map(exp.prompt.split()[0]), absolute=False)[b:-b, b:-b]
        intensity = heat_map2.cuda().mean().item()

        prompt_num_objects = num_map[exp.prompt.split()[0]]
        Xs[0].append(num_objects)
        ys[0].append(intensity)

        if num_objects == prompt_num_objects:  # Remove confounding effect of syntax, occlusion, etc.
            Xs[1].append(num_objects)
            ys[1].append(intensity)
        else:
            Xs[2].append(num_objects)
            ys[2].append(intensity)
            continue

        plt.clf()
        plot_overlay_heat_map(np.array(exp.image)[b:-b, b:-b], heat_map2)
        plt.show()

        # Nouns
        b = 1
        heat_map2 = expand_image(heat_map.compute_word_heat_map(exp.prompt.split()[-1]), absolute=False)[b:-b, b:-b]
        heat_map2 = (heat_map2 * 255).floor().cpu().numpy().astype(np.uint8)
        label_image = label(heat_map2 > 0.75 * 255)  # Use best threshold
        num_objects_pred = sum(rx.area > 200 for rx in regionprops(label_image))

        # TODO: apply same treatment to both nouns and numerals and then see if we can get some interesting (and contrasting) results
        # TODO: so reviewers don't think we're cherry-picking.
        # TODO: Hypotheses: (1) numerals represent the space between objects (2) nouns represent the objects themselves (3) nouns are more salient than numerals.

        if num_objects == prompt_num_objects or True:
            num_trues.append(num_objects)
            num_preds.append(num_objects_pred)
            num_prompts.append(prompt_num_objects)

    num_trues = np.array(num_trues)
    num_preds = np.array(num_preds)
    num_prompts = np.array(num_prompts)
    print(np.mean(num_preds == num_trues))
    print(np.mean(num_preds == num_prompts))

    print(len(Xs[1]), len(Xs[2]), len(Xs[1]) / (len(Xs[1]) + len(Xs[2])))
    print(scipy.stats.spearmanr(Xs[0], ys[0]))
    print(scipy.stats.spearmanr(Xs[1], ys[1]))
    print(scipy.stats.spearmanr(Xs[2], ys[2]))


if __name__ == '__main__':
    main()
