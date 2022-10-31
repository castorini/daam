# What the DAAM: Interpreting Stable Diffusion Using Cross Attention

**Caveat**: the codebase is in a bit of a mess. I plan to continue refactoring and polishing until it's published. Please contact me (Raphael Tang) if you have any questions.

In [our paper](https://arxiv.org/abs/2210.04885), we propose diffusion attentive attribution maps (DAAM), a cross attention-based approach for interpreting Stable Diffusion.
Check out our demo: https://huggingface.co/spaces/tetrisd/Diffusion-Attentive-Attribution-Maps.

## Using DAAM

I still have to package the codebase into a pip package.
For now, clone the repo and use DAAM directly, e.g.,

```python
from matplotlib import pyplot as plt

from daam.trace import trace
from daam.utils import expand_image, plot_overlay_heat_map 

# `pipe` is a StableDiffusionPipeline from the `diffusers` package
with trace(pipe) as trc:
    output = pipe(prompt)
    prompt = 'A dog and a cat'
    
    heat_map = trc.compute_word_heat_map('dog', prompt)
    heat_map = expand_image(heat_map)
    
    plot_overlay_heat_map(output.images[0], heat_map)
    plt.show()
```

## Running the Demo

To run the demo locally, execute this in an environment with PyTorch:
```bash
git clone https://github.com/castorini/daam && cd daam
cd space
pip install -r requirements.txt
huggingface-cli login  # Create an account and generate a token on huggingface.co
python app.py
```

Then, open http://127.0.0.1:8080 in your web browser.

Our datasets are here: https://git.uwaterloo.ca/r33tang/daam-data

## Citation
```
@article{tang2022daam,
  title={What the {DAAM}: Interpreting Stable Diffusion Using Cross Attention},
  author={Tang, Raphael and Pandey, Akshat and Jiang, Zhiying and Yang, Gefei and Kumar, Karun and Lin, Jimmy and Ture, Ferhan},
  journal={arXiv:2210.04885},
  year={2022}
}
```
