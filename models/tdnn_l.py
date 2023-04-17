import torch
import eval as d2l
import torch.nn as nn
# import torch.nn.functional as F
from models.tdnn_module import ECAPA_TDNN


class ECAPAModel(nn.Module):
    def __init__(self, n_class, C=1024, output_embedding=True, not_grad=False):
        super(ECAPAModel, self).__init__()
        self.in_features = 192
        self.output_num = n_class
        self.output_embedding = output_embedding
        self.speaker_encoder = ECAPA_TDNN(C=C)
        self.fc = nn.Linear(192, self.output_num)
        if not not_grad:
            for param in self.speaker_encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False

    def forward(self, x, aug=True):
        out = self.speaker_encoder(x, aug=aug)
        if not self.output_embedding:
            return self.fc(out)
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
    net = ECAPAModel(100, 1024, False)
    net.load_parameters("../pretrain.model", d2l.try_gpu())
    X = torch.zeros(2, 90000)
    output = net(X)
    print(output.shape)
    # parameters = torch.load("../pretrain.model", map_location=d2l.try_gpu())
    # print(parameters)
