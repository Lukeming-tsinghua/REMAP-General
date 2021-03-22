from EntityPairItem import BertEntityPairItem
from utils import set_all_seed
from data import BertEntityPairDataset
import joblib
import torch

set_all_seed(0)


pred_path = "../data/EntityPairItems-disodiso-gold-500-token-term.jl"
result_path = "result/proposal2/19/pred/result.pth"
case_path = "cases/proposal2-19-pred.txt"
sample_num = 12


if __name__ == "__main__":
    with open(pred_path, "rb") as f:
        data = joblib.load(f)

    result = torch.load(result_path, map_location="cpu")
    pred = result[1]
    label = result[2]
    with open(case_path, "w") as fo:
        for i, each in enumerate(data):
            p = pred[i]
            l = label[i]
            if p != l:
                samples = each.fetch(sample_num)
                cui1 = samples[1][0]
                cui2 = samples[1][1]
                text = samples[0][0]
                structure = samples[0][1]
                fo.write("[CUI1]: %s, [CUI2]: %s\n" % (cui1, cui2))
                for idx, (sent, stru) in enumerate(zip(text, structure)):
                    fo.write("[[%d]]\n[sent]:%s\n[struc]:%s\n" % (idx, sent, stru))
                fo.write("[PRED]: %d, [LABEL]: %d\n" % (p, l))
                fo.write("\n")
