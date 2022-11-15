from pathlib import Path
import argparse

from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import joblib

from daam import HeatMap, MmDetectHeatMap
from daam.experiment import GenerationExperiment
from daam.utils import cached_nlp, expand_image


def main():
    def run_mm_detect(path: Path):
        GenerationExperiment.clear_prediction_masks(path, args.prefix_name)
        heat_map = MmDetectHeatMap(path / '_masks.pred.mask2former.pt', threshold=args.threshold)

        for word, mask in heat_map.word_masks.items():
            GenerationExperiment.save_prediction_mask(path, mask, word, 'mmdetect')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--extract-types', '-e', type=str, nargs='+', default=['noun'])
    parser.add_argument('--model', '-m', type=str, default='daam', choices=['daam', 'mmdetect'])
    parser.add_argument('--threshold', '-t', type=float, default=0.4)
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--truth-only', action='store_true')
    parser.add_argument('--prefix-name', '-p', type=str, default='daam')
    parser.add_argument('--save-heat-map', action='store_true')
    args = parser.parse_args()

    extract_types = set(args.extract_types)
    model_id = 'CompVis/stable-diffusion-v1-4'
    tokenizer = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).tokenizer
    jobs = []

    for path in tqdm(list(Path(args.input_folder).glob('*'))):
        if not path.is_dir() or (not GenerationExperiment.contains_truth_mask(path) and args.truth_only):
            continue

        if list(path.glob('**/*.heat_map.png')) and args.save_heat_map:
            continue

        exp = GenerationExperiment.load(path)

        if args.model == 'daam':
            heat_map = HeatMap(tokenizer, exp.prompt, exp.global_heat_map)
            doc = cached_nlp(exp.prompt)

            for token in doc:
                if token.pos_.lower() in extract_types or 'all' in extract_types:
                    try:
                        word_heat_map = heat_map.compute_word_heat_map(token.text)
                    except:
                        continue

                    im = expand_image(word_heat_map, absolute=args.absolute, threshold=args.threshold)
                    exp.save_prediction_mask(im, token.text, args.prefix_name)

                    if args.save_heat_map:
                        exp.save_heat_map(tokenizer, token.text)

                    tqdm.write(f'Saved mask for {token.text} in {path}')
        else:
            jobs.append(joblib.delayed(run_mm_detect)(path))

    if jobs:
        joblib.Parallel(n_jobs=16)(tqdm(jobs))


if __name__ == '__main__':
    main()
