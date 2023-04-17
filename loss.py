import math
import torch
import eval as d2l
import torch.nn as nn
# import torch.nn.functional as F


def prec_accuracy(output, target, topk=(1,)):
    mask = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(mask, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = d2l.try_gpu()
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for _, (X, y) in enumerate(data_iter):
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
            phi = net(X)

            one_hot = torch.zeros(phi.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
            one_hot.scatter_(1, y.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * phi)
            prec = prec_accuracy(output.detach(), y.detach(), topk=(1,))[0]
            metric.add(prec * size(y), size(y))

    return metric[0] / metric[1]


class AAMSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=20, easy_margin=False):  # or margin=0.2, scale=30
        super(AAMSoftmax, self).__init__()
        self.test_normalize = True
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.output_num = nClasses
        # self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        # nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print('Initialised AAMSoftmax margin:%.3f scale:%.3f' % (self.m, self.s))

    def forward(self, cosine, label):
        assert cosine.size()[0] == label.size()[0]
        assert cosine.size()[1] == self.output_num
        # cos(theta)
        # cosine = F.linear(F.normalize(cosine), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros_like(cosine)
        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec = prec_accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec
