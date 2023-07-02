import numpy as np


def APFD(results_list, name2isfault):
    n = len(results_list)
    m = 0
    tf = 0
    for i, item in enumerate(results_list):
        if name2isfault[item]:
            tf += i
            m += 1
    return 1 - (tf / (n * m)) + (1 / (2 * n))


def POBL(results_list, name2isfault, ratio):
    n = len(results_list)
    m = 0
    cnt = 0
    for i in range(int(n)):
        if name2isfault[results_list[i]]:
            m += 1
            if i < ratio * n:
                cnt += 1
    return cnt / m


def RAUC(results_list, name2isfault):
    rat = [x for x in range(101)]
    y = []
    yb = []
    best_list = sorted(results_list, key=lambda x: name2isfault[x], reverse=True)
    for i in rat:
        y.append(POBL(results_list, name2isfault, i / 100.) * 100)
        yb.append(POBL(best_list, name2isfault, i / 100.) * 100)
    return np.trapz(y, rat) / np.trapz(yb, rat)

from sklearn.metrics import roc_auc_score

def ROC_AUC(results_list, name2isfault, score_list):
    y_true = [name2isfault[x] for x in results_list]
    return roc_auc_score(y_true,score_list)
