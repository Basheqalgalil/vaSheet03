#!/usr/bin/python2.7

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("probability_file")
parser.add_argument("label_file")
parser.add_argument("n_classes")
args = parser.parse_args()

n_classes = int(args.n_classes)

# generate prediction by averaging framewise scores
recognized = []
with open(args.probability_file, 'r') as f:
    content = f.read().split('\n')[2:-1]
    video_score = np.zeros((n_classes,))
    for line in content:
        if line == '#':
            recognized.append( video_score.argmax() )
            video_score = np.zeros((n_classes,))
        else:
            video_score += np.array( [ float(x) for x in line.split() ] )

# compare to ground truth
with open(args.label_file, 'r') as f:
    content = f.read().split('\n')[2:-1]
    ground_truth = [ int(line) for line in content ]

total = len(ground_truth)
correct = len([ i for i in range(total) if ground_truth[i] == recognized[i] ])
print 'Accuracy: %f' % (float(correct) / total)
