from nltk import FreqDist
from sklearn.metrics import f1_score
import argparse
import numpy as np
from scipy.spatial.distance import cosine


def read_conll_file(file):
    sents = []
    gold_labels = []
    predicted_labels = []

    sent = []
    gold = []
    pred = []
    for line in open(file):
        if line.strip() == "":
            sents.append(sent)
            gold_labels.append(gold)
            predicted_labels.append(pred)
            sent = []
            gold = []
            pred = []
        else:
            tok, g, pred1, pred2, pred3, pred4, pred5 = line.strip().split()
            sent.append(tok)
            gold.append(g)
            pred.append([pred1, pred2, pred3, pred4, pred5])
    return sents, gold_labels, predicted_labels

def majority_vote(preds):
    majority_preds = []
    for pred in preds:
        vote_pred = []
        for token in pred:
            majorty_label, count = FreqDist(token).most_common(1)[0]
            vote_pred.append(majorty_label)
        majority_preds.append(vote_pred)
    return majority_preds

def mtl_improvements(sents, golds, stl_preds_maj, mtl_preds_maj):
    total = 0
    improved = 0
    improved_examples = []

    for sent, gold, stl, mtl in zip(sents, golds, stl_preds_maj, mtl_preds_maj):
        total += 1

        stl_f1 = f1_score(gold, stl, average="macro")
        mtl_f1 = f1_score(gold, mtl, average="macro")

        if mtl_f1 > stl_f1:
            improved += 1
            improved_examples.append((sent, gold, stl, mtl))

    print("Number of dev examples improved by MTL: {0} / {1}".format(improved, total))
    print("Percentage of dev examples improved by MTL: {0:.1f}%".format((improved / total) * 100))
    return improved_examples, improved / total

def stl_improvements(sents, golds, stl_preds_maj, mtl_preds_maj):
    total = 0
    improved = 0
    improved_examples = []

    for sent, gold, stl, mtl in zip(sents, golds, stl_preds_maj, mtl_preds_maj):
        total += 1

        stl_f1 = f1_score(gold, stl, average="macro")
        mtl_f1 = f1_score(gold, mtl, average="macro")

        if stl_f1 > mtl_f1:
            improved += 1
            improved_examples.append((sent, gold, stl, mtl))

    print("Number of dev examples hurt by MTL: {0} / {1}".format(improved, total))
    print("Percentage of dev examples hurt by MTL: {0:.1f}%".format((improved / total) * 100))
    return improved_examples, improved / total

def negation_mtl_improvements(sents, golds, stl_preds_maj, mtl_preds_maj):
    negation_cues = set(["not", "doesnt", "n't", "cannot"])
    total = 0
    improved = 0
    improved_examples = []

    for sent, gold, stl, mtl in zip(sents, golds, stl_preds_maj, mtl_preds_maj):
        if len(negation_cues.intersection(set(sent))) > 0:
            total += 1

            stl_f1 = f1_score(gold, stl, average="macro")
            mtl_f1 = f1_score(gold, mtl, average="macro")

            if mtl_f1 > stl_f1:
                improved += 1
                improved_examples.append((sent, gold, stl, mtl))

    print("Number of negated dev examples improved by MTL: {0} / {1}".format(improved, total))
    print("Percentage of negated dev examples improved by MTL: {0:.1f}%".format((improved / total) * 100))
    return improved_examples, improved / total

def negation_stl_improvements(sents, golds, stl_preds_maj, mtl_preds_maj):
    negation_cues = set(["not", "doesnt", "n't", "cannot"])
    total = 0
    improved = 0
    improved_examples = []

    for sent, gold, stl, mtl in zip(sents, golds, stl_preds_maj, mtl_preds_maj):
        if len(negation_cues.intersection(set(sent))) > 0:
            total += 1

            stl_f1 = f1_score(gold, stl, average="macro")
            mtl_f1 = f1_score(gold, mtl, average="macro")

            if stl_f1 > mtl_f1:
                improved += 1
                improved_examples.append((sent, gold, stl, mtl))

    print("Number of negated dev examples hurt by MTL: {0} / {1}".format(improved, total))
    print("Percentage of negated dev examples hurt by MTL: {0:.1f}%".format((improved / total) * 100))
    return improved_examples, improved / total

def print_differences(improved_examples):
    row_format = "{:>15}" * 4
    for text, gold, stl, mtl in improved_examples:
        for i in zip(text, gold, stl, mtl):
            print(row_format.format(*i))
        print("-"*30)
        print()

def get_targets(sents, labels):
    targets = {}

    target = []
    lab = ""
    for i, (sent, label) in enumerate(zip(sents, labels)):
        for tok, l in zip(sent, label):
            if l is "O" and len(target) > 0:
                target = " ".join(target)
                if target not in targets:
                    targets[target] = {"NEG": 0, "POS": 0, "NEU": 0, "Count": 0}
                try:
                    targets[target][lab] += 1
                    targets[target]["Count"] += 1
                except KeyError:
                    pass
                target = []
                lab = ""
            elif l.startswith("B") or l.startswith("U"):
                target.append(tok)
                lab = l.split("-")[-1]
            elif l.startswith("I") or l.startswith("L"):
                target.append(tok)
            else:
                pass

    targets = [(k, v) for k, v in targets.items()]
    targets = sorted(targets, key=lambda i: i[1]["Count"], reverse=True)
    return targets

def convert_to_array(distribution):
    neg = distribution["NEG"]
    pos = distribution["POS"]
    neu = distribution["NEU"]
    return np.array([neg, pos, neu])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", default="data/results/en/stl/restaurant/dev.conll")
    parser.add_argument("--mtl", default="data/results/en/mtl/conan_doyle/restaurant/dev.conll")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    stl_sents, stl_gold, stl_preds = read_conll_file(args.stl)
    mtl_sents, mtl_gold, mtl_preds = read_conll_file(args.mtl)

    stl_preds_maj = majority_vote(stl_preds)
    mtl_preds_maj = majority_vote(mtl_preds)

    # find and show differences between STL and MTL models

    # which examples do MTL models improve on?
    print("-" * 80)
    print("Multitask improved")
    print("-" * 80)
    mtl_imp, mtl_count = mtl_improvements(stl_sents, stl_gold, stl_preds_maj, mtl_preds_maj)
    if args.verbose == True:
        print_differences(mtl_imp)
    print("-" * 40)
    # which examples do MTL models perform worse on?
    stl_imp, stl_count = stl_improvements(stl_sents, stl_gold, stl_preds_maj, mtl_preds_maj)
    if args.verbose == True:
        print_differences(stl_imp)
    print()
    print()

    # for the subset of examples with negation cues, what differences?
    print("-" * 80)
    print("Negated subsection")
    print("-" * 80)
    neg_mtl_imp, neg_mtl_count = negation_mtl_improvements(stl_sents, stl_gold, stl_preds_maj, mtl_preds_maj)
    print("-" * 40)
    neg_stl_imp, neg_stl_count = negation_stl_improvements(stl_sents, stl_gold, stl_preds_maj, mtl_preds_maj)
    if args.verbose == True:
        print_differences(neg_mtl_imp)
    print()
    print()

    # find all targets and label distribution of each model
    gold_targets = get_targets(stl_sents, stl_gold)
    stl_targets = get_targets(stl_sents, stl_preds_maj)
    mtl_targets = get_targets(stl_sents, mtl_preds_maj)

    if args.verbose == True:
        print("-" * 80)
        print("10 most common targets for each model")
        print("-" * 80)
        print("GOLD", end=": ")
        print(", ".join([i[0] for i in gold_targets[:10]]))
        print("STL", end=": ")
        print(", ".join([i[0] for i in stl_targets[:10]]))
        print("MTL", end=": ")
        print(", ".join([i[0] for i in mtl_targets[:10]]))

    # In order to figure out, in general, which targets stl and mtl perform worst on, we can look at the polarity distribution over them and see if these are very different from the gold standard.

    gold = dict([(i[0], i[1]) for i in gold_targets])
    stl = dict([(i[0], i[1]) for i in stl_targets])
    mtl = dict([(i[0], i[1]) for i in mtl_targets])

    stl_dist = 0
    mtl_dist = 0

    stl_outliers = []
    mtl_outliers = []

    for target, distr in gold.items():
        g = convert_to_array(distr)
        try:
            s = convert_to_array(stl[target])
            dist = cosine(g, s)
            stl_dist += dist
            if dist > 0.4:
                print("STL---")
                print(g)
                print(s)
                print(dist)
                print("-------")
                stl_outliers.append(target)
        except KeyError:
            stl_dist += 1
        try:
            m = convert_to_array(mtl[target])
            dist = cosine(g, m)
            mtl_dist += dist
            if dist > 0.4:
                print("MTL---")
                print(g)
                print(m)
                print(dist)
                print("-------")
                mtl_outliers.append(target)
        except KeyError:
            mtl_dist += 1

    print("Mean cosine distance of target distributions for STL: {0}".format(stl_dist / len(gold)))
    print("Mean cosine distance of target distributions for MTL: {0}".format(mtl_dist / len(gold)))
