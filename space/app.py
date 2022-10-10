from threading import Lock
import math
import os
import random

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import get_global_heat_map, clear_heat_maps
from matplotlib import pyplot as plt
import gradio as gr
import torch
import torch.nn.functional as F
import spacy


if not os.environ.get('NO_DOWNLOAD_SPACY'):
    spacy.cli.download('en_core_web_sm')


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

gen = torch.Generator(device='cuda')
gen.manual_seed(12758672)
orig_state = gen.get_state()
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
lock = Lock()
nlp = spacy.load('en_core_web_sm')


def expand_m(m, n: int = 1, o=512, mode='bicubic'):
    m = m.unsqueeze(0).unsqueeze(0) / n
    m = F.interpolate(m.float().detach(), size=(o, o), mode='bicubic', align_corners=False)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = m.cpu().detach()

    return m


@torch.no_grad()
def predict(prompt, threshold):
    global lock
    with torch.cuda.amp.autocast(), lock:
        gen.set_state(orig_state.clone())
        clear_heat_maps()

        out = pipe(prompt, guidance_scale=7.5, height=512, width=512, do_intermediates=False, generator=gen, num_inference_steps=50)
        heat_maps = get_global_heat_map()

    with torch.cuda.amp.autocast(dtype=torch.float32):
        m = 0
        n = 0
        w = ''
        w_idx = 0

        fig, ax = plt.subplots()
        ax.imshow(out.images[0].cpu().float().detach().permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])

        fig1, axs1 = plt.subplots(math.ceil(len(out.words) / 4), 4)#, figsize=(20, 20))
        fig2, axs2 = plt.subplots(math.ceil(len(out.words) / 4), 4)  # , figsize=(20, 20))

        for idx in range(len(out.words) + 1):
            if idx == 0:
                continue

            word = out.words[idx - 1]
            m += heat_maps[idx]
            n += 1
            w += word

            if '</w>' not in word:
                continue
            else:
                mplot = expand_m(m, n)
                spotlit_im = out.images[0].cpu().float().detach()
                w = w.replace('</w>', '')
                spotlit_im2 = torch.cat((spotlit_im, (1 - mplot.squeeze(0)).pow(1)), dim=0)

                if len(out.words) <= 4:
                    a1 = axs1[w_idx % 4]
                    a2 = axs2[w_idx % 4]
                else:
                    a1 = axs1[w_idx // 4, w_idx % 4]
                    a2 = axs2[w_idx // 4, w_idx % 4]

                a1.set_xticks([])
                a1.set_yticks([])
                a1.imshow(mplot.squeeze().numpy(), cmap='jet')
                a1.imshow(spotlit_im2.permute(1, 2, 0).numpy())
                a1.set_title(w)

                mask = torch.ones_like(mplot)
                mask[mplot < threshold * mplot.max()] = 0
                im2 = spotlit_im * mask.squeeze(0)
                a2.set_xticks([])
                a2.set_yticks([])
                a2.imshow(im2.permute(1, 2, 0).numpy())
                a2.set_title(w)
                m = 0
                n = 0
                w_idx += 1
                w = ''

        for idx in range(w_idx, len(axs1.flatten())):
            fig1.delaxes(axs1.flatten()[idx])
            fig2.delaxes(axs2.flatten()[idx])

    return fig, fig1, fig2


with gr.Blocks() as demo:
    md = '''# DAAM: Attention Maps for Interpreting Stable Diffusion
    Check out the paper: [test](http://google.com)
    
    Try `steak and dollars on a plate` or `a fox, a dog, and a wolf in a field`. Can you figure out what went wrong?
    '''
    gr.Markdown(md)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Prompt')
            slider = gr.Slider(0, 1.0, value=0.4, interactive=True, step=0.05, label='Threshold (tau)')
            submit_btn = gr.Button('Submit')

        with gr.Tab('Original Image'):
            p0 = gr.Plot()

        with gr.Tab('Soft DAAM Maps'):
            p1 = gr.Plot()

        with gr.Tab('Hard DAAM Maps'):
            p2 = gr.Plot()
        
        submit_btn.click(fn=predict, inputs=[text, slider], outputs=[p0, p1, p2])


demo.launch(server_name='0.0.0.0', server_port=8080)

