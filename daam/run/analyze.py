from functools import lru_cache
from pathlib import Path
import argparse
import json
import random

from tqdm import trange, tqdm
from torch import autocast
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy

from diffusers.models.attention import get_global_heat_map, clear_heat_maps


class SpatialAttributePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(32, 32)

    def compute_heatmap(self, image_x, target_x, idx: int = 1):
        x = self.forward(image_x)
        x = x[0]
        heat_map = torch.zeros(x.shape[0], x.shape[1]).cuda()
        for i in trange(x.size(0), position=0):
            for j in trange(x.size(1), position=1):
                target_x.grad = None
                x[i, j].backward(retain_graph=True)
                # grad_norm = target_x.grad[1, idx].norm().item()
                grad_norm = F.relu(-((target_x.grad[1, idx] * target_x[1, idx])).sum()).item()
                heat_map[i, j] = grad_norm
                tqdm.write(str((i, j, grad_norm)))

            # from matplotlib import pyplot as plt
            # plt.imshow(heat_map.cpu().numpy())
            # plt.show()

        return heat_map

    def forward(self, x):
        x = self.pool(x)
        return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def expand_m(m, n: int = 1, o=512, abs_: bool = False):
    m = m.unsqueeze(0).unsqueeze(0) / n
    m = F.interpolate(m.float().detach(), size=(o, o), mode="bicubic", align_corners=False)
    # m[m < 0.5 * m.max()] = 0
    # print(m.max())
    if not abs_:
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = m.cpu().detach()

    return m


def shrink_im(im, n: int = 64):
    return F.interpolate(im.unsqueeze(0).unsqueeze(0), size=(n, n), mode="bicubic", align_corners=False).squeeze(0).squeeze(0)


nlp = None


@lru_cache(maxsize=100000)
def nlp_cache(prompt: str, type='en_core_web_md'):
    global nlp
    if nlp is None:
        nlp = spacy.load(type)
    return nlp(prompt)


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    torch.no_grad().__enter__()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='prompt', choices=['prompt', 'caption'])
    parser.add_argument('--output-folder', '-o', type=str, default='output')
    parser.add_argument('--save-full', action='store_true')
    parser.add_argument('--save-intermediates', action='store_true')
    parser.add_argument('--num-gen', type=int, default=1000)
    parser.add_argument('--split', type=str, default='coco', choices=['coco', 'unreal'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    plt.rcParams.update({'font.size': 36})

    with torch.cuda.amp.autocast(dtype=torch.float16):

        output_folder = Path(args.output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda"

        gen = torch.Generator(device='cuda')
        gen.manual_seed(args.seed)
        orig_state = gen.get_state()
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, generator=gen)
        pipe = pipe.to(device)
        sa_pool = SpatialAttributePool().cuda().eval()

        for p in dir(pipe):
            attr = getattr(pipe, p)

            if isinstance(attr, nn.Module):
                for p in attr.parameters():
                    p.requires_grad = False

        if args.action == 'caption':
            with open('annotations/captions_val2014.json') as f:
                captions = json.load(f)['annotations']

            random.shuffle(captions)

            nouns = []
            adjectives = []

            for caption in tqdm(captions[:5000]):
                doc = nlp_cache(caption['caption'])
                nouns.extend([token.text for token in doc if token.pos_ == 'NOUN'])
                adjectives.extend([token.text for token in doc if token.pos_ == 'ADJ'])

            captions = iter(captions)

        num_gen = 0

        while True:
            num_gen += 1
            gen.set_state(orig_state.clone())

            if args.num_gen == num_gen:
                break

            try:
                if args.action == 'prompt':
                    prompt = input('> ')
                    id = 'test'
                else:
                    c = next(captions)
                    prompt = c['caption']

                    if args.split == 'unreal':
                        doc = nlp_cache(prompt)

                        for token in doc:
                            if token.pos_ == 'NOUN':
                                prompt = prompt.replace(token.text, random.choice(nouns))
                            elif token.pos_ == 'ADJ':
                                prompt = prompt.replace(token.text, random.choice(adjectives))

                        print(prompt)

                    id = str(c['image_id'])

                with autocast("cuda"):
                    # latents, text = pipe.intermediate_forward(prompt, guidance_scale=5, height=448, width=448)
                    # latents = latents.detach()
                    # text = text.detach()
                    # text.requires_grad = True
                    # text.retain_grad()
                    # out = pipe.final_step(text, latents, guidance_scale=5)
                    clear_heat_maps()
                    plt.clf()
                    out = pipe(prompt, guidance_scale=7.5, height=512, width=512, do_intermediates=args.save_intermediates, generator=gen)
                    of = output_folder / id

                    of.mkdir(exist_ok=True, parents=True)
                    sp_im = out.images[0].cpu().float().detach()
                    plt.imshow(sp_im.permute(1, 2, 0).numpy())
                    plt.savefig((of / "_orig.png").__str__())
                    out.pil_images[0].save((of / "_orig_annotate.png").__str__())

                    if args.save_intermediates:
                        for idx, im in enumerate(out.intermediates):
                            plt.clf()
                            plt.imshow(im.permute(1, 2, 0).cpu().float().detach().numpy())
                            plt.savefig((of / f"_intermediate.{idx}.png").__str__())

                    with (of / "prompt.txt").open('w') as f:
                        print(prompt, file=f)

                    if args.save_intermediates:
                        continue

                    # heat_map = sa_pool.compute_heatmap(out.images, out.text_embeddings)
                    factors = set()

                    for f in (8, 4, 2):
                        factors.add(f)
                        heat_map = get_global_heat_map(factors=factors)
                        heat_map_loc = get_global_heat_map(factors={f})

                        with torch.cuda.amp.autocast(dtype=torch.float32):
                            heat_map = heat_map.detach()
                            heat_map_loc = heat_map_loc.detach()
                            m = 0
                            m_loc = 0
                            n = 0
                            w = ''
                            w_idx = 0

                            doc = nlp_cache(prompt)

                            for idx in range(len(out.words) + 1):
                                if idx == 0:
                                    continue

                                word = out.words[idx - 1]
                                m += heat_map[idx]
                                m_loc += heat_map_loc[idx]
                                n += 1
                                w += word

                                if '</w>' not in word:
                                    continue
                                else:
                                    mplot = expand_m(m, n)
                                    m_locplot = expand_m(m_loc, n)
                                    viz = 'all'  # 'spotlight' or 'heat' or 'all

                                    facs = '-'.join(sorted(list(map(str, factors))))
                                    plt.clf()

                                    # print(doc[w_idx].pos_)

                                    if True or doc[w_idx].pos_ == 'NOUN' or doc[w_idx].pos_ == 'PROPN':# or doc[w_idx].pos_ == 'ADJ':
                                        spotlit_im = out.images[0].cpu().float().detach()
                                        w = w.replace('</w>', '')
                                        torch.save(m, (of / f"{w}.{w_idx}.attrmap.{facs}.pt").__str__())
                                        torch.save(m_loc, (of / f"{w}.{w_idx}.attrmaploc.{f}.pt").__str__())

                                        if viz == 'heat' or viz == 'all':
                                            spotlit_im2 = torch.cat((spotlit_im, (1 - mplot.squeeze(0)).pow(1)), dim=0)
                                            plt.imshow(mplot.squeeze().numpy(), cmap='jet')
                                            plt.imshow(spotlit_im2.permute(1, 2, 0).numpy())
                                            plt.title(w)
                                            plt.savefig((of / f"{w}.{facs}.{w_idx}.heat.png").__str__())
                                        if viz == 'spotlight' or viz == 'all':
                                            spotlit_im2 = spotlit_im * mplot.squeeze(0)
                                            # spotlit_im = torch.cat((spotlit_im, (1 - m.squeeze(0)).pow(1)), dim=0)
                                            plt.imshow(spotlit_im2.permute(1, 2, 0).numpy())
                                            plt.title(w)
                                            plt.savefig((of / f"{w}.{facs}.{w_idx}.spotlight.png").__str__())
                                        if viz == 'mask' or viz == 'all':
                                            mask = torch.ones_like(mplot)
                                            mask[mplot < 0.4 * mplot.max()] = 0
                                            im2 = spotlit_im * mask.squeeze(0)
                                            plt.imshow(im2.permute(1, 2, 0).numpy())
                                            plt.title(w)
                                            plt.savefig((of / f"{w}.{facs}.{w_idx}.mask.png").__str__())
                                    m = 0
                                    m_loc = 0
                                    n = 0
                                    w_idx += 1
                                    w = ''

                    # print(heat_map
            except ZeroDivisionError:
                pass



if __name__ == '__main__':
    main()
