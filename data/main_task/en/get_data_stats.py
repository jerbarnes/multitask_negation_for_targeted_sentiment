import argparse
import numpy as np

def read_conll(file):
    sents, targets, labels = [], [], []
    sent, targ, label = [], [], []
    current_targ = []
    for line in open(file):
        if line.strip() == "":
            if current_targ != []:
                targ.append(current_targ)
            if sent != []:
                sents.append(sent)
                targets.append(targ)
                labels.append(label)
            sent, targ, label = [], [], []
        else:
            tok, l = line.strip().split()
            sent.append(tok)
            if l[0] == "B":
                bio, pol = l.split("-")
                current_targ.append(tok)
                label.append(pol)
            elif l[0] == "I":
                current_targ.append(tok)
            elif l[0] == "L":
                current_targ.append(tok)
                targ.append(current_targ)
                current_targ = []
            elif l[0] == "U":
                bio, pol = l.split("-")
                current_targ.append(tok)
                label.append(pol)
                targ.append(current_targ)
                current_targ = []
    return sents, targets, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="laptop/train.conll")

    args = parser.parse_args()

    sents, targets, labels = read_conll(args.file)

    print()
    # Number of Targets
    num_targets = len([i for j in targets for i in j])
    print("Number of Targets: {0}".format(num_targets))

    # Average Length of Targets
    lengths = [len(i) for j in targets for i in j]
    ave = np.mean(lengths)
    print("Avg. Target Length: {0:.1f}".format(ave))

    # Number of sents with multiple conflicting polarities
    count = 0
    for sent in labels:
        if len(set(sent)) > 1:
            count += 1
    print("Number of Multiple Polarities: {0}".format(count))
    print()
