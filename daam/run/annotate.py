from pathlib import Path
import argparse

from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt

from daam import GenerationExperiment, HeatMap, plot_overlay_heat_map, expand_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--pred-prefix', '-p', type=str, default='daam')
    parser.add_argument('--model', type=str, default='CompVis/stable-diffusion-v1-4')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, use_auth_token=True)
    tokenizer = pipe.tokenizer
    del pipe

    for path in input_folder.iterdir():
        if not path.is_dir():
            continue

        exp = GenerationExperiment.load(str(path), args.pred_prefix)

        if exp.annotations is not None and 'num_objects' in exp.annotations:
            continue

        plt.clf()
        print(exp.prompt)
        exp.image.show()

        heat_map = HeatMap(tokenizer, exp.prompt, exp.global_heat_map)
        plt.clf()
        plot_overlay_heat_map(exp.image, expand_image(heat_map.compute_word_heat_map(exp.prompt.split()[0])))
        plt.show()

        num_objects = int(input('Number of objects: '))
        exp.annotate('num_objects', num_objects)
        exp.save_annotations()


if __name__ == '__main__':
    main()
