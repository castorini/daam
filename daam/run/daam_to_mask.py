from pathlib import Path
import argparse

from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from daam import HeatMap
from daam.experiment import GenerationExperiment
from daam.utils import cached_nlp, expand_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--extract-types', '-e', type=str, nargs='+', default=['noun'])
    parser.add_argument('--threshold', '-t', type=float, default=0.4)
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--truth-only', action='store_true')
    parser.add_argument('--prefix-name', '-p', type=str, default='daam')
    args = parser.parse_args()

    extract_types = set(args.extract_types)
    model_id = 'CompVis/stable-diffusion-v1-4'
    tokenizer = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).tokenizer

    for path in tqdm(Path(args.input_folder).glob('*')):
        if not path.is_dir() or (not GenerationExperiment.contains_truth_mask(path) and args.truth_only):
            continue

        exp = GenerationExperiment.load(path)
        heat_map = HeatMap(tokenizer, exp.prompt, exp.global_heat_map)
        doc = cached_nlp(exp.prompt)

        for token in doc:
            if token.pos_.lower() in extract_types:
                try:
                    word_heat_map = heat_map.compute_word_heat_map(token.text)
                except:
                    continue

                im = expand_image(word_heat_map, absolute=args.absolute, threshold=args.threshold)
                exp.save_prediction_mask(im, token.text, args.prefix_name)

                tqdm.write(f'Saved mask for {token.text} in {path}')


if __name__ == '__main__':
    main()
