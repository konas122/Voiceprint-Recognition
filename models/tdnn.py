import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from models.tdnn_module import PreEmphasis, FbankAug


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim, context):
        super().__init__()
        self.context = context
        if self.context:
            in_dims = in_dim * 3
        else:
            in_dims = in_dim
        self.linear = nn.Sequential(
            nn.Conv1d(in_dims, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        t = x.size()[-1]
        if self.context:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t),
                ),
                dim=1,
            )
        else:
            global_x = x
        alpha = self.linear(global_x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192, output_num=10,
                 context=True, aug=True, embedding=True):
        super().__init__()
        self.context = context
        self.aug = aug
        self.embedding = embedding
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.fbank = torch.nn.Sequential(
            PreEmphasis(),
            T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80)
        )
        self.specaug = FbankAug()  # Spec augmentation

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128, self.context)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

        self.weight = torch.nn.Parameter(torch.FloatTensor(output_num, embd_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.fbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if self.aug:
                x = self.specaug(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out1 + out2)
        out4 = self.layer4(out1 + out2 + out3)

        # out1 = self.layer1(x)
        # out2 = self.layer2(out1) + out1
        # out3 = self.layer3(out1 + out2) + out1 + out2
        # out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        if out.shape[0] == 1:
            out = self.linear(self.pooling(out))
        else:
            out = self.bn1(self.pooling(out))
            out = self.bn2(self.linear(out))

        if not self.embedding:
            return F.linear(F.normalize(out), F.normalize(self.weight))
        return out


if __name__ == '__main__':
    X = torch.zeros(2, 90000)
    model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192, output_num=10, context=True, embedding=False)
    output = model(X)
    # print(model)
    print(output.shape)  # [2, 192] or [2, output_num]
