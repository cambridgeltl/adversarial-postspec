#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'ivan'

import sys,os, codecs, operator
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np

in_we_file = ""
reps = {}
SYN = 50

# Parse the input word embedding (WE) file
def parse_we_file(in_we_file):

    source_dict = {}

    with open(in_we_file, "r") as in_file:
        lines = in_file.readlines()

    in_file.close()
    # traverse the lines and
    print >> sys.stderr, 'Loading and normalizing word embeddings... ' + os.path.basename(in_we_file)
    # input vectors, but skip the first two lines (only numbers)
    for i in xrange(2,len(lines)):
        try:
            temp_list = lines[i].split()
            dkey = temp_list.pop(0)
            x = np.array(temp_list, dtype='double')
            norm = np.linalg.norm(x)
            source_dict[dkey] = x/norm
        except (RuntimeError, IndexError, ValueError, UnicodeDecodeError):
            continue

    return source_dict

def nn_compute(word1):
    w1_score_dict = {}
    for target_word in reps:
        # Skip the same word
        if word1 == target_word:
            continue
        w1_score_dict[target_word] = 1-np.inner(reps[word1],reps[target_word])
        # Now sort the dictionary of scores based on their distances
    w1_sorted_score_dict = sorted(w1_score_dict.items(), key=operator.itemgetter(1))[:SYN]

    print w1_sorted_score_dict


# Entry point for the entire program
if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Incorrect usage. Usage: python %s in_rep_file in_word' % sys.argv[0])

    in_we_file = sys.argv[1]
    in_word  = str(sys.argv[2]).lower()
    reps = parse_we_file(in_we_file)

    nn_compute(in_word)

