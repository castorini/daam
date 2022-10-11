# [What the DAAM!?](https://arxiv.org/abs/2210.04885)

We propose diffusion attentive attribution maps (DAAM), a cross attention-based approach for interpreting Stable Diffusion.
Check out our demo: http://daam.ralphtang.com:8080.

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
