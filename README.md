# What the DAAM: Interpreting Stable Diffusion Using Cross Attention

[![HF Spaces](https://img.shields.io/badge/HuggingFace%20Space-online-green.svg)](https://huggingface.co/spaces/tetrisd/Diffusion-Attentive-Attribution-Maps) [![Citation](https://img.shields.io/badge/Citation-ACL-orange.svg)](https://gist.github.com/daemon/639de6fea584d7df1a62f04a2ea0cdad) [![PyPi version](https://badgen.net/pypi/v/daam?color=blue)](https://pypi.org/project/daam) [![Downloads](https://static.pepy.tech/badge/daam)](https://pepy.tech/project/daam)

![example image](example.jpg)

### Updated to support Stable Diffusion XL (SDXL) and Diffusers 0.21.1!

I regularly update this codebase. Please submit an issue if you have any questions.

In [our paper](https://aclanthology.org/2023.acl-long.310), we propose diffusion attentive attribution maps (DAAM), a cross attention-based approach for interpreting Stable Diffusion.
Check out our demo: https://huggingface.co/spaces/tetrisd/Diffusion-Attentive-Attribution-Maps.
See our [documentation](https://castorini.github.io/daam/), hosted by GitHub pages, and [our Colab notebook](https://colab.research.google.com/drive/1miGauqa07uHnDoe81NmbmtTtnupmlipv?usp=sharing), updated for v0.1.0.

## Getting Started
First, install [PyTorch](https://pytorch.org) for your platform.
Then, install DAAM with `pip install daam`, unless you want an editable version of the library, in which case do `git clone https://github.com/castorini/daam && pip install -e daam`.
Finally, login using `huggingface-cli login` to get many stable diffusion models -- you'll need to get a token at [HuggingFace.co](https://huggingface.co/).

### Running the Website Demo
Simply run `daam-demo` in a shell and navigate to http://localhost:8080.
The same demo as the one on HuggingFace Spaces will show up.

### Using DAAM as a CLI Utility
DAAM comes with a simple generation script for people who want to quickly try it out.
Try running
```bash
$ mkdir -p daam-test && cd daam-test
$ daam "A dog running across the field."
$ ls
a.heat_map.png    field.heat_map.png    generation.pt   output.png  seed.txt
dog.heat_map.png  running.heat_map.png  prompt.txt
```
Your current working directory will now contain the generated image as `output.png` and a DAAM map for every word, as well as some auxiliary data.
You can see more options for `daam` by running `daam -h`.
To use Stable Diffusion XL as the backend, run `daam --model xl-base-1.0 "Dog jumping"`.  

### Using DAAM as a Library

Import and use DAAM as follows:

```python
from daam import trace, set_seed
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch


model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'

pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
pipe = pipe.to(device)

prompt = 'A dog runs across the field'
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=50, generator=gen)
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('dog')
        heat_map.plot_overlay(out.images[0])
        plt.show()
```

You can also serialize and deserialize the DAAM maps pretty easily:

```python
from daam import GenerationExperiment, trace

with trace(pipe) as tc:
    pipe('A dog and a cat')
    exp = tc.to_experiment('experiment-dir')
    exp.save()  # experiment-dir now contains all the data and heat maps

exp = GenerationExperiment.load('experiment-dir')  # load the experiment
```

We'll continue adding docs.
In the meantime, check out the `GenerationExperiment`, `GlobalHeatMap`, and `DiffusionHeatMapHooker` classes, as well as the `daam/run/*.py` example scripts.
You can download the COCO-Gen dataset from the paper at http://ralphtang.com/coco-gen.tar.gz.
If clicking the link doesn't work on your browser, copy and paste it in a new tab, or use a CLI utility such as `wget`.

## See Also
- [DAAM-i2i](https://github.com/RishiDarkDevil/daam-i2i), an extension of DAAM to image-to-image attribution.

- [Furkan's video](https://www.youtube.com/watch?v=XiKyEKJrTLQ) on easily getting started with DAAM.

- [1littlecoder's video](https://www.youtube.com/watch?v=J2WtkA1Xfew) for a code demonstration and Colab notebook of an older version of DAAM.

## Citation
```
@inproceedings{tang2023daam,
    title = "What the {DAAM}: Interpreting Stable Diffusion Using Cross Attention",
    author = "Tang, Raphael  and
      Liu, Linqing  and
      Pandey, Akshat  and
      Jiang, Zhiying  and
      Yang, Gefei  and
      Kumar, Karun  and
      Stenetorp, Pontus  and
      Lin, Jimmy  and
      Ture, Ferhan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
    url = "https://aclanthology.org/2023.acl-long.310",
}
```
