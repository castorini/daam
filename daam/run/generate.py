from collections import defaultdict
from pathlib import Path
import argparse
import json
import random
import sys
import time

import pandas as pd
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import inflect
import numpy as np
import torch

from daam import trace
from daam.experiment import GenerationExperiment, build_word_list_coco80
from daam.utils import set_seed, cached_nlp, auto_device, auto_autocast


def main():
    actions = ['quickgen', 'prompt', 'coco', 'template', 'cconj', 'coco-unreal', 'stdin', 'regenerate']
    model_id_map = {
        'v1': 'runwayml/stable-diffusion-v1-5',
        'v2-base': 'stabilityai/stable-diffusion-2-base',
        'v2-large': 'stabilityai/stable-diffusion-2',
        'v2-1-base': 'stabilityai/stable-diffusion-2-1-base',
        'v2-1-large': 'stabilityai/stable-diffusion-2-1'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', nargs='?', type=str)
    parser.add_argument('--action', '-a', type=str, choices=actions, default=actions[0])
    parser.add_argument('--low-memory', action='store_true')
    parser.add_argument('--model', type=str, default='v2-1-base', choices=list(model_id_map.keys()))
    parser.add_argument('--output-folder', '-o', type=str)
    parser.add_argument('--input-folder', '-i', type=str, default='input')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gen-limit', type=int, default=1000)
    parser.add_argument('--template', type=str, default='{numeral} {noun}')
    parser.add_argument('--template-data-file', '-tdf', type=str, default='template.tsv')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--num-timesteps', '-n', type=int, default=30)
    parser.add_argument('--all-heads', action='store_true')
    parser.add_argument('--word', type=str)
    parser.add_argument('--random-seed', action='store_true')
    parser.add_argument('--truth-only', action='store_true')
    parser.add_argument('--save-heads', action='store_true')
    parser.add_argument('--load-heads', action='store_true')
    args = parser.parse_args()

    eng = inflect.engine()
    args.lemma = cached_nlp(args.word)[0].lemma_ if args.word else None
    model_id = model_id_map[args.model]
    seeds = []

    if args.action.startswith('coco'):
        with (Path(args.input_folder) / 'captions_val2014.json').open() as f:
            captions = json.load(f)['annotations']

        random.shuffle(captions)
        new_captions = []

        if args.action == 'coco-unreal':
            pos_map = defaultdict(list)

            for caption in tqdm(captions):
                doc = cached_nlp(caption['caption'])

                for tok in doc:
                    if tok.pos_ == 'ADJ' or tok.pos_ == 'NOUN':
                        pos_map[tok.pos_].append(tok)

            for caption in tqdm(captions):
                doc = cached_nlp(caption['caption'])
                new_tokens = []

                for tok in doc:
                    if tok.pos_ == 'ADJ' or tok.pos_ == 'NOUN':
                        new_tokens.append(random.choice(pos_map[tok.pos_]))

                new_prompt = ''.join([tok.text_with_ws for tok in new_tokens])
                caption['caption'] = new_prompt

                print(new_prompt)

            new_prompt = ''.join([tok.text_with_ws for tok in new_tokens])
            caption['caption'] = new_prompt

            print(new_prompt)
            new_captions.append(caption)

        prompts = [(caption['id'], caption['caption']) for caption in captions]
    elif args.action == 'stdin':
        prompts = []

        for idx, line in enumerate(sys.stdin):
            prompts.append((idx, line.strip()))
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
        words_map = build_word_list_coco80()
        prompts = []

        for idx in range(args.gen_limit):
            use_cohyponym = random.random() < 0.5

            if use_cohyponym:
                c = random.choice(list(words_map.keys()))
                w1, w2 = np.random.choice(words_map[c], 2, replace=False)
            else:
                c1, c2 = np.random.choice(list(words_map.keys()), 2, replace=False)
                w1 = random.choice(words_map[c1])
                w2 = random.choice(words_map[c2])

            prompt_id = f'{"cohypo" if use_cohyponym else "diff"}-{idx}'
            a1 = 'an' if w1[0] in 'aeiou' else 'a'
            a2 = 'an' if w2[0] in 'aeiou' else 'a'
            prompt = f'{a1} {w1} and {a2} {w2}'
            prompts.append((prompt_id, prompt))
    elif args.action == 'quickgen':
        if args.output_folder is None:
            args.output_folder = '.'

        prompts = [('.', args.prompt)]
    elif args.action == 'regenerate':
        prompts = []

        for exp_folder in Path(args.input_folder).iterdir():
            if not GenerationExperiment.contains_truth_mask(exp_folder) and args.truth_only:
                continue

            prompt = GenerationExperiment.read_prompt(exp_folder)
            prompts.append((exp_folder.name, prompt))
            seeds.append(GenerationExperiment.read_seed(exp_folder))

        if args.output_folder is None:
            args.output_folder = args.input_folder
    else:
        prompts = [('prompt', input('> '))]

    if args.output_folder is None:
        args.output_folder = 'output'

    new_prompts = []

    if args.lemma is not None:
        for prompt_id, prompt in tqdm(prompts):
            if args.lemma not in prompt.lower():
                continue

            doc = cached_nlp(prompt)
            found = False

            for tok in doc:
                if tok.lemma_.lower() == args.lemma and not found:
                    found = True
                elif tok.lemma_.lower() == args.lemma:  # filter out prompts with multiple instances of the word
                    found = False
                    break

            if found:
                new_prompts.append((prompt_id, prompt))

        prompts = new_prompts

    prompts = prompts[:args.gen_limit]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = auto_device(pipe)

    with auto_autocast(dtype=torch.float16), torch.no_grad():
        for gen_idx, (prompt_id, prompt) in enumerate(tqdm(prompts)):
            seed = int(time.time()) if args.random_seed else args.seed
            prompt = prompt.replace(',', ' ,').replace('.', ' .').strip()

            if seeds and gen_idx < len(seeds):
                seed = seeds[gen_idx]

            gen = set_seed(seed)

            if args.action == 'cconj':
                seed = int(prompt_id.split('-')[1]) + args.seed_offset
                gen = set_seed(seed)

            prompt_id = str(prompt_id)

            with trace(pipe, low_memory=args.low_memory, save_heads=args.save_heads, load_heads=args.load_heads) as tc:
                out = pipe(prompt, num_inference_steps=args.num_timesteps, generator=gen, callback=tc.time_callback)
                exp = tc.to_experiment(args.output_folder, id=prompt_id, seed=seed)
                exp.save(args.output_folder, heat_maps=args.action == 'quickgen')

                if args.all_heads:
                    exp.clear_checkpoint()

                for word in prompt.split():
                    if args.lemma is not None and cached_nlp(word)[0].lemma_.lower() != args.lemma:
                        continue

                    exp.save_heat_map(word)

                    if args.all_heads:
                        for head_idx in range(16):
                            for layer_idx, layer_name in enumerate(tc.layer_names):
                                try:
                                    heat_map = tc.compute_global_heat_map(layer_idx=layer_idx, head_idx=head_idx)
                                    exp = GenerationExperiment(
                                        path=Path(args.output_folder),
                                        id=prompt_id,
                                        global_heat_map=heat_map.heat_maps,
                                        seed=seed,
                                        prompt=prompt,
                                        image=out.images[0]
                                    )

                                    exp.save_heat_map(word, output_prefix=f'l{layer_idx}-{layer_name}-h{head_idx}-')
                                except RuntimeError:
                                    print(f'Missing ({layer_idx}, {head_idx}, {layer_name})')


if __name__ == '__main__':
    main()
