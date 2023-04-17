import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tdnn_module import ECAPA_TDNN as TDNN


class Pretrain_TDNN(nn.Module):
    def __init__(self, n_class, C=1024, output_embedding=True, not_grad=False, aug=True):
        super(Pretrain_TDNN, self).__init__()
        self.aug = aug
        self.in_features = 192
        self.output_num = n_class
        self.output_embedding = output_embedding
        self.speaker_encoder = TDNN(C=C)
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        if not not_grad:
            for param in self.speaker_encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.speaker_encoder(x, aug=self.aug)
        if not self.output_embedding:
            return F.linear(F.normalize(out), F.normalize(self.weight))
        else:
            return out

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path, device):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=device)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


if __name__ == '__main__':
    net = Pretrain_TDNN(100, 1024, True)
    X = torch.zeros(2, 41500)
    output = net(X)
    print(output.shape)
