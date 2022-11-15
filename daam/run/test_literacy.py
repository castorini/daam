from functools import cache
from pathlib import Path
import argparse

from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

from daam import GenerationExperiment, COCO80_LABELS, COCO80_INDICES


def main():
    @torch.no_grad()
    @cache
    def compute_cosine_dist(word1, word2):
        text_encoder.eval()
        text_input = tokenizer(
            f'a {word1} and a {word2}',
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )

        text_embeddings = text_encoder(text_input.input_ids.cuda())[0][0]
        a = text_embeddings[text_input.input_ids.squeeze().tolist().index(537) - 1]  # 537 is ID for 'and'
        b = text_embeddings[text_input.input_ids.squeeze().tolist().index(49407) - 1]  # 49407 is ID for padding

        return torch.cosine_similarity(a, b, dim=0).item()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--pred-prefix', '-p', type=str, default='mmdetect')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    n = len(COCO80_LABELS)

    word1_present_matrix = np.zeros((n, n), dtype=np.int32)
    word2_present_matrix = np.zeros((n, n), dtype=np.int32)
    corr_matrix = np.zeros((n, n), dtype=np.int32)
    tot_matrix = np.zeros((n, n), dtype=np.int32)
    tot_subset_matrix = np.zeros((n, n), dtype=np.int32)
    cos_matrix = np.zeros((n, n), dtype=np.float32)

    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    del pipe

    for path in tqdm(list(input_folder.iterdir())):
        if not path.is_dir():
            continue

        exp = GenerationExperiment.load(str(path), args.pred_prefix)
        word1, word2 = exp.prompt.split(' and ')
        word1 = word1.strip()[2:]
        word2 = word2.strip()[2:]
        lbl1 = COCO80_INDICES[word1]
        lbl2 = COCO80_INDICES[word2]

        if exp.nsfw():
            continue

        compute_cosine_dist(word1, word2)

        if args.visualize:
            print(exp.prompt, word1 in exp.prediction_masks, word2 in exp.prediction_masks)
            plt.clf()
            exp.image.show()
            plt.show()

        word1_present_matrix[lbl1, lbl2] += word1 in exp.prediction_masks
        word2_present_matrix[lbl1, lbl2] += word2 in exp.prediction_masks
        corr_matrix[lbl1, lbl2] += word1 in exp.prediction_masks and word2 in exp.prediction_masks
        tot_subset_matrix[lbl1, lbl2] += (word1 in exp.prediction_masks) or (word2 in exp.prediction_masks)
        tot_matrix[lbl1, lbl2] += 1
        cos_matrix[lbl1, lbl2] = compute_cosine_dist(word1, word2)

    torch.save((word1_present_matrix, word2_present_matrix, corr_matrix, tot_matrix, tot_subset_matrix, cos_matrix), 'results2.pt')


if __name__ == '__main__':
    main()
