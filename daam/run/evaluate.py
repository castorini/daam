from pathlib import Path
import argparse

from tqdm import tqdm

from daam.evaluate import MeanEvaluator, UnsupervisedEvaluator
from daam.experiment import GenerationExperiment, COCOSTUFF27_LABELS, COCO80_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, required=True)
    parser.add_argument('--pred-prefix', '-p', type=str, default='daam')
    parser.add_argument('--mask-type', '-m', type=str, default='word', choices=['word', 'composite'])
    parser.add_argument('--eval-type', '-e', type=str, default='labeled', choices=['labeled', 'unlabeled', 'hungarian'])
    parser.add_argument('--restrict-set', '-r', type=str, default='none', choices=['none', 'coco27', 'coco80'])
    parser.add_argument('--subtype', '-st', type=str, default='.')
    args = parser.parse_args()

    evaluator = MeanEvaluator() if args.eval_type != 'hungarian' else UnsupervisedEvaluator()
    simplify80 = False
    vocab = []

    if args.restrict_set == 'coco27':
        simplify80 = True
        vocab = COCOSTUFF27_LABELS
    elif args.restrict_set == 'coco80':
        vocab = COCO80_LABELS

    if not vocab:
        for path in tqdm(Path(args.input_folder).glob('*')):
            if not path.is_dir() or not GenerationExperiment.contains_truth_mask(path):
                continue

            exp = GenerationExperiment.load(
                path,
                args.pred_prefix,
                composite=args.mask_type == 'composite',
                simplify80=simplify80
            )

            vocab.extend(exp.truth_masks)
            vocab.extend(exp.prediction_masks)

        vocab = list(set(vocab))
        vocab.sort()

    for path in tqdm(Path(args.input_folder).glob('*')):
        if not path.is_dir() or not GenerationExperiment.contains_truth_mask(path):
            continue

        exp = GenerationExperiment.load(
            path,
            args.pred_prefix,
            composite=args.mask_type == 'composite',
            simplify80=simplify80,
            vocab=vocab,
            subtype=args.subtype
        )

        if args.eval_type == 'labeled':
            for word, mask in exp.truth_masks.items():
                if word not in vocab and args.restrict_set != 'none':
                    continue

                try:
                    evaluator.log_iou(exp.prediction_masks[word], mask)
                    evaluator.log_intensity(exp.prediction_masks[word])
                except KeyError:
                    continue
        elif args.eval_type == 'hungarian':
            for gt_word, gt_mask in exp.truth_masks.items():
                if gt_word not in vocab and args.restrict_set != 'none':
                    continue

                for pred_word, pred_mask in exp.prediction_masks.items():
                    try:
                        evaluator.log_iou(pred_mask, gt_mask, vocab.index(gt_word), vocab.index(pred_word))
                    except (KeyError, ValueError):
                        continue

                evaluator.increment()
        else:
            for word, mask in exp.truth_masks.items():
                evaluator.log_iou(list(exp.prediction_masks.values()), mask)

    print(evaluator)


if __name__ == '__main__':
    main()
