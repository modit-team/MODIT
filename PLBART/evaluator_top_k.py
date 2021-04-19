# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys

from bleu_top_k import _bleu
import numpy as np


def calculate_scores(references, predictions, topk):
    length = len(references)
    count = 0
    for i in range(length):
        r = references[i]
        p = predictions[i]
        for j in range(topk):
            if p[j] == r:
                count += 1
                break
    acc = count / length * 100
    bleu_score = _bleu(references, predictions[:, :topk].tolist())
    return acc, bleu_score
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--references', '-ref', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-pre', help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--beam', help='Max Beam Size', default=10, type=int)
    parser.add_argument('--nbest', help='Number of Top K', nargs='+', type=int, default=[1, 2, 5, 10])

    args = parser.parse_args()

    refs = [x.strip() for x in open(args.references, 'r', encoding='utf-8').readlines()]
    pres = [x.strip() for x in open(args.predictions, 'r', encoding='utf-8').readlines()]

    predictions = []
    for i in range(len(refs)):
        predictions.append(pres[(i*args.beam):((i+1)*args.beam)])
        pass
    predictions = np.array(predictions)
    for topk in args.nbest:
        acc, bleu = calculate_scores(refs, predictions, topk)
        print('TopK: %d\tAccuracy: %.2f\tBLEU: %.2f' % (topk, acc, bleu))
        pass


if __name__ == '__main__':
    main()
