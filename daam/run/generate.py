from pathlib import Path
import argparse
import json
import random

from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import torch

from daam import trace
from daam.experiment import GenerationExperiment
from daam.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='prompt', choices=['prompt', 'coco'])
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--input-folder', '-i', type=str, default='input')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gen-limit', type=int, default=1000)
    args = parser.parse_args()

    gen = set_seed(args.seed)

    if args.action == 'coco':
        with (Path(args.input_folder) / 'captions_val2014.json').open() as f:
            captions = json.load(f)['annotations'][:args.gen_limit]

        random.shuffle(captions)
        prompts = [(caption['id'], caption['caption']) for caption in captions]
    else:
        prompts = [('prompt', input('> '))]

    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)

    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        for prompt_id, prompt in tqdm(prompts):
            with trace(pipe, weighted=True) as tc:
                out = pipe(prompt, num_inference_steps=30, generator=gen)
                exp = GenerationExperiment(
                    id=str(prompt_id),
                    global_heat_map=tc.compute_global_heat_map(prompt).heat_maps,
                    seed=args.seed,
                    prompt=prompt,
                    image=out.images[0]
                )
                exp.save(args.output_folder)


if __name__ == '__main__':
    main()
