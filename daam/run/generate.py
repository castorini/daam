from collections import defaultdict
from pathlib import Path
import argparse
import json
import pandas as pd
import random

from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import inflect
import torch

from daam import trace
from daam.experiment import GenerationExperiment
from daam.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='prompt', choices=['prompt', 'coco', 'template', 'cconj'])
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--input-folder', '-i', type=str, default='input')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gen-limit', type=int, default=1000)
    parser.add_argument('--template', type=str, default='{numeral} {noun}')
    parser.add_argument('--scramble-unreal', action='store_true')
    parser.add_argument('--template-data-file', '-tdf', type=str, default='template.tsv')
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()

    gen = set_seed(args.seed)
    eng = inflect.engine()

    if args.action == 'coco':
        with (Path(args.input_folder) / 'captions_val2014.json').open() as f:
            captions = json.load(f)['annotations']

        random.shuffle(captions)
        captions = captions[:args.gen_limit]
        prompts = [(caption['id'], caption['caption']) for caption in captions]
    elif args.action == 'template':
        template_df = pd.read_csv(args.template_data_file, sep='\t', quoting=3)
        sample_dict = defaultdict(list)

        for name, df in template_df.groupby('pos'):
            sample_dict[name].extend(df['word'].tolist())

        prompts = []
        template_words = args.template.split()
        plural_numerals = {'0', '2', '3', '4', '5', '6', '7', '8', '9', 'zero', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}

        for prompt_id in range(args.gen_limit):
            words = []
            pluralize = False

            for word in template_words:
                if word.startswith('{'):
                    pos = word[1:-1]
                    word = random.choice(sample_dict[pos])

                    if pos == 'noun' and pluralize:
                        word = eng.plural(word)

                words.append(word)
                pluralize = word in plural_numerals

            prompt_id = str(prompt_id)
            prompts.append((prompt_id, ' '.join(words)))
            tqdm.write(str(prompts[-1]))
    elif args.action == 'cconj':
        template_df = pd.read_csv(args.template_data_file, sep='\t', quoting=3)
        sample_dict = defaultdict(list)

        for name, df in template_df.groupby('pos'):
            sample_dict[name].extend(df['word'].tolist())

        prompts = []
        prompt_id = 0

        for _ in range(args.gen_limit):

            for word1 in tqdm(sample_dict['noun']):
                for word2 in sample_dict['noun']:
                    if word1 == word2:
                        continue

                    prompt = f'a {word1} and a {word2}'
                    print(prompt)

                    prompts.append((str(prompt_id), prompt))
                    prompt_id += 1
    else:
        prompts = [('prompt', input('> '))]

    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)
    seed = args.seed

    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        for prompt_id, prompt in tqdm(prompts):
            if args.action == 'template' or args.action == 'cconj':
                gen = set_seed(int(prompt_id))
                seed = int(prompt_id)

            prompt_id = str(prompt_id)

            if args.regenerate and not GenerationExperiment.contains_truth_mask(args.output_folder, prompt_id):
                print(f'Skipping {prompt_id}')
                continue

            with trace(pipe, weighted=True) as tc:
                out = pipe(prompt, num_inference_steps=20, generator=gen)
                exp = GenerationExperiment(
                    id=prompt_id,
                    global_heat_map=tc.compute_global_heat_map(prompt).heat_maps,
                    seed=seed,
                    prompt=prompt,
                    image=out.images[0]
                )
                exp.save(args.output_folder)


if __name__ == '__main__':
    main()
