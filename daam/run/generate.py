from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import argparse
import json

import pandas as pd
import random

from diffusers import StableDiffusionPipeline
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import inflect
import numpy as np
import torch

from daam import trace
from daam.experiment import GenerationExperiment, build_word_list_coco80
from daam.utils import set_seed, cached_nlp


def build_word_list_large() -> Dict[str, List[str]]:
    cat5 = ['vegetable', 'fruit', 'car', 'mammal', 'reptile']
    topk = open('data/top100k').readlines()[:30000]
    topk = set(w.strip() for w in topk)
    words_map = {}

    for cat in cat5:
        words = set()
        x = wn.synsets(cat, 'n')[0]
        hyponyms = list(x.closure(lambda s: s.hyponyms()))

        for synset in hyponyms:
            if any('_' in w for w in synset.lemma_names()):
                continue

            word = synset.lemma_names()[0].lower()

            if '_' not in word and word in topk:
                words.add(word)

        words_map[cat] = list(words)

    return words_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='prompt', choices=['prompt', 'coco', 'template', 'cconj', 'coco-unreal'])
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--input-folder', '-i', type=str, default='input')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gen-limit', type=int, default=1000)
    parser.add_argument('--template', type=str, default='{numeral} {noun}')
    parser.add_argument('--scramble-unreal', action='store_true')
    parser.add_argument('--template-data-file', '-tdf', type=str, default='template.tsv')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0)
    args = parser.parse_args()

    gen = set_seed(args.seed)
    eng = inflect.engine()

    if args.action.startswith('coco'):
        with (Path(args.input_folder) / 'captions_val2014.json').open() as f:
            captions = json.load(f)['annotations']

        random.shuffle(captions)
        captions = captions[:args.gen_limit]
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
    else:
        prompts = [('prompt', input('> '))]

    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)
    seed = args.seed

    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        for prompt_id, prompt in tqdm(prompts):
            # gen = set_seed(seed)  # Uncomment this for seed fix

            if args.action == 'template' or args.action == 'cconj':
                seed = int(prompt_id.split('-')[1]) + args.seed_offset
                gen = set_seed(seed)

            prompt_id = str(prompt_id)
            do_skip = False
            num_steps = 30

            if args.regenerate and not GenerationExperiment.contains_truth_mask(args.output_folder, prompt_id):
                # I screwed up with the seed generation so this is a hacky workaround to reproduce the paper.
                num_steps = 1
                do_skip = True
                print(f'Skipping {prompt_id}')
            elif args.regenerate:
                print(f'Regenerating {prompt_id}')

            with trace(pipe, weighted=False) as tc:
                out = pipe(prompt, num_inference_steps=num_steps, generator=gen)
                exp = GenerationExperiment(
                    id=prompt_id,
                    global_heat_map=tc.compute_global_heat_map(prompt).heat_maps,
                    seed=seed,
                    prompt=prompt,
                    image=out.images[0]
                )

                if not do_skip:
                    exp.save(args.output_folder)


if __name__ == '__main__':
    main()
