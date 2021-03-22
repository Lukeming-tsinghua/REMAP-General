import numpy as np
import pickle
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == "__main__":
    relation_num = 8
    for e in range(0, 300, 10):
        pres = torch.load("result/proposal-bce-i2b2/"+str(e)+"/pred/result.pth")
        pscore = np.vstack(pres[0])
        ptrue = np.array(pres[2])

        ## calculate mrr
        prank = np.argsort(-pscore, axis=1)
        rank = []
        for i in range(len(ptrue)):
            rank.append(np.where(ptrue[i]==prank[i, :])[0])
        rank = np.array(rank) + 1
        mrr = np.mean(1/rank)
        print("mrr:",mrr)

        ## calculate P@N
        pmax = np.max(pscore, axis=1)
        pindex = np.argsort(-pmax)
        N = [100, 300, 500]
        for n in N:
            pindexn = pindex[:n]
            ptruen = ptrue[pindexn]
            pscoren = pscore[pindexn].argmax(axis=1)
            acc = accuracy_score(pscoren, ptruen)
            print("P@%d:%.4f"% (n, acc))
