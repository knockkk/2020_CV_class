import torch
from torch.nn import functional as F


def focal_loss(gamma, preds, labels):
    preds = preds.view(-1, preds.size(-1))

    preds_softmax = F.softmax(preds, dim=1)
    preds_logsoft = torch.log(preds_softmax)

    # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
    preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1)) #挑出最大的概率
    # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
    loss = -torch.mul(torch.pow((1-preds_softmax), gamma), preds_logsoft) #很好理解，如果容易区分的对象，那么Loss很小

    loss = loss.mean()

    return loss
