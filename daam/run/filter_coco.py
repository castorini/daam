from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import re
import sys

from nltk.stem import PorterStemmer
from tqdm import tqdm

from daam.experiment import build_word_list_coco80


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', type=str, default='input')
    parser.add_argument('--limit', '-lim', type=int, default=500)
    args = parser.parse_args()

    with (Path(args.input_folder) / 'captions_val2014.json').open() as f:
        captions = json.load(f)['annotations']

    vocab = build_word_list_coco80()
    stemmer = PorterStemmer()
    words = set(stemmer.stem(w) for items in vocab.values() for w in items)
    word_patt = '(' + '|'.join(words) + ')'
    patt = re.compile(rf'^.*(?P<word1>{word_patt}) and (a )?(?P<word2>{word_patt}).*$')

    c = Counter()
    data = defaultdict(list)

    for caption in tqdm(captions):
        sentence = caption['caption'].split()
        sentence = ' '.join(stemmer.stem(w) for w in sentence)
        match = patt.match(sentence)

        if match:
            word1 = match.groupdict()['word1']
            word2 = match.groupdict()['word2']
            print(f'{word1} and {word2} found', file=sys.stderr)

            words = tuple(sorted([word1, word2]))
            c[words] += 1
            data[words].append(caption)

    all_captions = []
    final_captions = []

    for words, count in c.most_common():
        all_captions.append(data[words])

    while all_captions:
        for captions in all_captions:
            if captions:
                final_captions.append(captions.pop(-1))

        idx = 0

        while idx < len(all_captions):
            if not all_captions[idx]:
                all_captions.pop(idx)
            else:
                idx += 1

    for captions in final_captions:
        print(json.dumps(captions))


if __name__ == '__main__':
    main()
