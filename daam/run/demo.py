import math
import time
from threading import Lock
from typing import Any, List
import argparse

import numpy as np
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import gradio as gr
import torch
from spacy import displacy

from daam import trace
from daam.utils import set_seed, cached_nlp, auto_autocast


def dependency(text):
    doc = cached_nlp(text)
    svg = displacy.render(doc, style='dep', options={'compact': True, 'distance': 100})

    return svg


def get_tokenizing_mapping(prompt: str, tokenizer: Any) -> List[List[int]]:
    tokens = tokenizer.tokenize(prompt)
    merge_idxs = []
    words = []
    curr_idxs = []
    curr_word = ''

    for i, token in enumerate(tokens):
        curr_idxs.append(i + 1)  # because of the [CLS] token
        curr_word += token
        if '</w>' in token:
            merge_idxs.append(curr_idxs)
            curr_idxs = []
            words.append(curr_word[:-4])
            curr_word = ''

    return merge_idxs, words


def get_args():
    model_id_map = {
        'v1': 'runwayml/stable-diffusion-v1-5',
        'v2-base': 'stabilityai/stable-diffusion-2-base',
        'v2-large': 'stabilityai/stable-diffusion-2',
        'v2-1-base': 'stabilityai/stable-diffusion-2-1-base',
        'v2-1-large': 'stabilityai/stable-diffusion-2-1',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='v2-1-base', choices=list(model_id_map.keys()), help="which diffusion model to use")
    parser.add_argument('--seed', '-s', type=int, default=0, help="the random seed")
    parser.add_argument('--port', '-p', type=int, default=8080, help="the port to launch the demo")
    parser.add_argument('--no-cuda', action='store_true', help="Use CPUs instead of GPUs")
    args = parser.parse_args()
    args.model = model_id_map[args.model]
    return args


def main():
    args = get_args()
    plt.switch_backend('agg')

    device = "cpu" if args.no_cuda else "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(args.model, use_auth_token=True).to(device)
    lock = Lock()

    @torch.no_grad()
    def update_dropdown(prompt):
        tokens = [''] + [x.text for x in cached_nlp(prompt) if x.pos_ == 'ADJ']
        return gr.Dropdown.update(choices=tokens), dependency(prompt)

    @torch.no_grad()
    def plot(prompt, choice, replaced_word, inf_steps, is_random_seed):
        new_prompt = prompt.replace(',', ', ').replace('.', '. ')

        if choice:
            if not replaced_word:
                replaced_word = '.'

            new_prompt = [replaced_word if tok.text == choice else tok.text for tok in cached_nlp(prompt)]
            new_prompt = ' '.join(new_prompt)

        merge_idxs, words = get_tokenizing_mapping(prompt, pipe.tokenizer)
        with auto_autocast(dtype=torch.float16), lock:
            try:
                plt.close('all')
                plt.clf()
            except:
                pass

            seed = int(time.time()) if is_random_seed else args.seed
            gen = set_seed(seed)
            prompt = prompt.replace(',', ', ').replace('.', '. ')  # hacky fix to address later

            if choice:
                new_prompt = new_prompt.replace(',', ', ').replace('.', '. ')  # hacky fix to address later

                with trace(pipe, save_heads=new_prompt != prompt) as tc:
                    out = pipe(prompt, num_inference_steps=inf_steps, generator=gen)
                    image = np.array(out.images[0]) / 255
                    heat_map = tc.compute_global_heat_map()

                if new_prompt == prompt:
                    image2 = image
                else:
                    gen = set_seed(seed)

                    with trace(pipe, load_heads=True) as tc:
                        out2 = pipe(new_prompt, num_inference_steps=inf_steps, generator=gen)
                        image2 = np.array(out2.images[0]) / 255
            else:
                with trace(pipe) as tc:
                    out = pipe(prompt, num_inference_steps=inf_steps, generator=gen)
                    image = np.array(out.images[0]) / 255
                    heat_map = tc.compute_global_heat_map()

        # the main image
        if new_prompt == prompt:
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)

            if choice:
                ax[1].imshow(image2)

            ax[0].set_title(choice)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_title(replaced_word)
            ax[1].set_xticks([])
            ax[1].set_yticks([])

        # the heat maps
        num_cells = 4
        w = int(num_cells * 3.5)
        h = math.ceil(len(words) / num_cells * 4.5)
        fig_soft, axs_soft = plt.subplots(math.ceil(len(words) / num_cells), num_cells, figsize=(w, h))
        axs_soft = axs_soft.flatten()
        with torch.cuda.amp.autocast(dtype=torch.float32):
            for idx, parsed_map in enumerate(heat_map.parsed_heat_maps()):
                word_ax_soft = axs_soft[idx]
                word_ax_soft.set_xticks([])
                word_ax_soft.set_yticks([])
                parsed_map.word_heat_map.plot_overlay(out.images[0], ax=word_ax_soft)
                word_ax_soft.set_title(parsed_map.word_heat_map.word, fontsize=12)

        for idx in range(len(words), len(axs_soft)):
            fig_soft.delaxes(axs_soft[idx])

        return fig, fig_soft

    with gr.Blocks(css='scrollbar.css') as demo:
        md = '''# DAAM: Attention Maps for Interpreting Stable Diffusion
        Check out the paper: [What the DAAM: Interpreting Stable Diffusion Using Cross Attention](http://arxiv.org/abs/2210.04885).
        See our (much cleaner) [DAAM codebase](https://github.com/castorini/daam) on GitHub.
        '''
        gr.Markdown(md)

        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown([
                    'An angry, bald man doing research',
                    'A bear and a moose',
                    'A blue car driving through the city',
                    'Monkey walking with hat',
                    'Doing research at Comcast Applied AI labs',
                    'Professor Jimmy Lin from the modern University of Waterloo',
                    'Yann Lecun teaching machine learning on a green chalkboard',
                    'A brown cat eating yummy cake for her birthday',
                    'A brown fox, a white dog, and a blue wolf in a green field',
                ], label='Examples', value='An angry, bald man doing research')

                text = gr.Textbox(label='Prompt', value='An angry, bald man doing research')

                with gr.Row():
                    doc = cached_nlp('An angry, bald man doing research')
                    tokens = [''] + [x.text for x in doc if x.pos_ == 'ADJ']
                    dropdown2 = gr.Dropdown(tokens, label='Adjective to replace', interactive=True)
                    text2 = gr.Textbox(label='New adjective', value='')

                checkbox = gr.Checkbox(value=False, label='Random seed')
                slider1 = gr.Slider(15, 30, value=25, interactive=True, step=1, label='Inference steps')

                submit_btn = gr.Button('Submit', elem_id='submit-btn')
                viz = gr.HTML(dependency('An angry, bald man doing research'), elem_id='viz')

            with gr.Column():
                with gr.Tab('Images'):
                    p0 = gr.Plot()

                with gr.Tab('DAAM Maps'):
                    p1 = gr.Plot()

            text.change(fn=update_dropdown, inputs=[text], outputs=[dropdown2, viz])
            
            submit_btn.click(
                fn=plot,
                inputs=[text, dropdown2, text2, slider1, checkbox],
                outputs=[p0, p1])
            dropdown.change(lambda prompt: prompt, dropdown, text)
            dropdown.update()

    while True:
        try:
            demo.launch(server_name='0.0.0.0', server_port=args.port)
        except OSError:
            gr.close_all()
        except KeyboardInterrupt:
            gr.close_all()
            break


if __name__ == '__main__':
    main()
