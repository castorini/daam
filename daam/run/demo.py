import math
from threading import Lock
from typing import Any, List

import numpy as np
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import gradio as gr
import torch

from daam import trace
from daam.utils import set_seed, expand_image


def get_tokenizing_mapping(prompt: str, tokenizer: Any) -> List[List[int]]:
    tokens = tokenizer.tokenize(prompt)
    merge_idxs = []
    words = []
    curr_idxs = []
    curr_word = ''

    for i, token in enumerate(tokens):
        curr_idxs.append(i+1)  # because of the [CLS] token
        curr_word += token
        if '</w>' in token:
            merge_idxs.append(curr_idxs)
            curr_idxs = []
            words.append(curr_word[:-4])
            curr_word = ''

    return merge_idxs, words


def main():
    # argparse?

    model_id = "stabilityai/stable-diffusion-2-base"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True,
    ).to(device)
    lock = Lock()

    @torch.no_grad()
    def predict(prompt, inf_steps, threshold):
        merge_idxs, words = get_tokenizing_mapping(prompt, pipe.tokenizer)
        with torch.cuda.amp.autocast(dtype=torch.float16), lock:
            try:
                plt.close('all')
            except:
                pass

            gen = set_seed(0)
            with trace(pipe) as tc:  # low memory?
                out = pipe(prompt, num_inference_steps=inf_steps, generator=gen)
                image = np.array(out.images[0]) / 255
                global_heat_maps = tc.compute_global_heat_map(prompt).heat_maps

        with torch.cuda.amp.autocast(dtype=torch.float32):
            # the main image
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

            # the heat maps
            fig_soft, axs_soft = plt.subplots(math.ceil(len(words) / 4), 4)
            fig_hard, axs_hard = plt.subplots(math.ceil(len(words) / 4), 4)
            axs_soft = axs_soft.flatten()
            axs_hard = axs_hard.flatten()

            for w_idx, word in enumerate(words):
                word_heat_map = expand_image(
                    global_heat_maps[merge_idxs[w_idx]].mean(0),
                    absolute=False,
                    out=image.shape[0])

                word_ax_soft = axs_soft[w_idx]
                word_ax_hard = axs_hard[w_idx]

                # soft
                word_ax_soft.set_xticks([])
                word_ax_soft.set_yticks([])
                spotlit_image = np.concatenate((
                    image, 1 - word_heat_map.unsqueeze(-1)
                ), axis=-1)
                word_ax_soft.imshow(word_heat_map, cmap='jet')
                word_ax_soft.imshow(spotlit_image)
                word_ax_soft.set_title(word, fontsize=12)
                
                # hard
                word_ax_hard.set_xticks([])
                word_ax_hard.set_yticks([])
                mask = np.ones_like(word_heat_map)
                mask[word_heat_map < threshold * word_heat_map.max()] = 0
                hard_masked_image = image * np.expand_dims(mask, axis=2)
                word_ax_hard.imshow(hard_masked_image)
                word_ax_hard.set_title(word, fontsize=12)

        for idx in range(len(words), len(axs_soft)):
            fig_soft.delaxes(axs_soft[idx])
            fig_hard.delaxes(axs_hard[idx])

        return fig, fig_soft, fig_hard
    

    with gr.Blocks() as demo:
        md = '''# DAAM: Attention Maps for Interpreting Stable Diffusion
        Check out the paper: [What the DAAM: Interpreting Stable Diffusion Using Cross Attention](http://arxiv.org/abs/2210.04885). Note that, due to server costs, this demo will transition to HuggingFace Spaces on 2022-10-20.
        '''
        gr.Markdown(md)

        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown([
                    'Cat playing with dog',
                    'An angry, bald man doing research',
                    'Doing research at Comcast Applied AI labs',
                    'Professor Jimmy Lin from the University of Waterloo',
                    'Yann Lecun teaching machine learning on a chalkboard',
                    'A cat eating cake for her birthday',
                    'Steak and dollars on a plate',
                    'A fox, a dog, and a wolf in a field',
                ], label='Examples', value='Cat playing with dog')

                text = gr.Textbox(label='Prompt', value='Cat playing with dog')
                slider1 = gr.Slider(15, 35, value=20, interactive=True, step=1, label='Inference steps')
                slider2 = gr.Slider(0, 1.0, value=0.5, interactive=True, step=0.05, label='Threshold for the hard maps')
                submit_btn = gr.Button('Submit')

            with gr.Tab('Original Image'):
                p0 = gr.Plot()

            with gr.Tab('Soft DAAM Maps'):
                p1 = gr.Plot()

            with gr.Tab('Hard DAAM Maps'):
                p2 = gr.Plot()
            
            submit_btn.click(
                fn=predict,
                inputs=[text, slider1, slider2],
                outputs=[p0, p1, p2])
            dropdown.change(lambda prompt: prompt, dropdown, text)
            dropdown.update()

    while True:
        try:
            demo.launch(server_name='0.0.0.0', server_port=8080)
        except OSError:
            gr.close_all()
        except KeyboardInterrupt:
            gr.close_all()
            break


if __name__ == '__main__':
    main()
