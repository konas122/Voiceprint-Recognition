import torch

import eval as d2l
from tools import eval_net
# from d2l import torch as d2l
from models.tdnn_pretrain import Pretrain_TDNN

if __name__ == "__main__":
    model_path = './param.model'
    Device = d2l.try_gpu()

    model2 = Pretrain_TDNN(420, 1024, False, not_grad=False)
    model2.load_parameters(model_path, Device)
    # model2 = torch.load('net.pth')

    EER, minDCF = eval_net(model2, Device, 10, 10)
    print(f'EER:{EER:.4f} minDCF:{minDCF:.4f}')
