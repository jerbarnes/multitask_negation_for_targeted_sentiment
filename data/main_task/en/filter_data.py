import argparse

def read_conll(file):
    sents, labels = [], []
    sent, label = [], []
    for line in open(file):
        if line.strip() == "":
            if len(sent) > 0:
                sents.append(sent)
                labels.append(label)
                sent, label = [], []
        else:
            tok, lab = line.strip().split()
            sent.append(tok)
            label.append(lab)
    return sents, labels


def filter_sents(sents, labels, cues):
    filtered_sents, filtered_labels = [], []
    for sent, label in zip(sents, labels):
        lower_cased_sent = " ".join(sent).lower().split()
        if len(cues.intersection(set(lower_cased_sent))) > 0:
            filtered_sents.append(sent)
            filtered_labels.append(label)
    return filtered_sents, filtered_labels


def print_to_conll(sents, labels, outfile):
    with open(outfile, "w") as out:
        for sent, label in zip(sents, labels):
            for tok, lab in zip(sent, label):
                out.write("{0} {1}\n".format(tok, lab))
            out.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="restaurant/test_neg.conll")
    parser.add_argument("--outfile", default="restaurant/test_neg_only.conll")

    neg_cues = set(['no', 'not', "n't", 'none', 'nothing', 'never', 'neither', 'without', "won't", "can't", 'nor', 'cannot', 'none'])

    spec_cues = set(['would', 'could', "'d", 'maybe', 'if', 'or', 'should', 'seems', 'might', 'wish', 'think', 'probably', 'supposedly', 'must', 'seemingly', 'perhaps', 'maybe', 'likely', 'guess', 'suspect', 'hope', 'believe', 'apparently'])

    args = parser.parse_args()

    if "neg" in args.file:
        cues = neg_cues
    else:
        cues = spec_cues


    sents, labels = read_conll(args.file)

    fsents, flabels = filter_sents(sents, labels, cues)

    print_to_conll(fsents, flabels, args.outfile)
