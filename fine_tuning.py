import torch
import loader
import train as t
import eval as d2l
# import torch_directml
from loss import AAMSoftmax
# from d2l import torch as d2l
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.tdnn_pretrain import Pretrain_TDNN


def load_model(path, output_num, device, not_grad=False):
    load_net = torch.load(path, map_location=device)
    model = Pretrain_TDNN(output_num, 1024, output_embedding=False, not_grad=not_grad)
    model.speaker_encoder = load_net.speaker_encoder
    del load_net
    return model


if __name__ == "__main__":
    people_num, data_per_people = 420, 10
    noise, mel, reverse = False, True, False
    margin, scale, easy_margin = 0.2, 20, False
    num_epochs, learn_rate, weight_decay = 40, 0.1, 1e-3
    learn_rate_period, learn_rate_decay = 10, 0.95
    mode, model_name = "train", "resnet18"
    hidden_size, num_layers = 64, 2

    # Device = torch_directml.device()
    # prefetch_factor, batch_size, num_works, persistent = 2, 32, 8, False

    Device = d2l.try_gpu()
    if Device.type == 'cpu':
        prefetch_factor, batch_size, num_works, persistent = 2, 8, 8, False
    elif torch.cuda.is_available():
        prefetch_factor, batch_size, num_works, persistent = 8, 256, 32, True
    else:
        prefetch_factor, batch_size, num_works, persistent = 2, 32, 8, False

    t.init_logs()
    train_dict, test_dict, people_num = loader.load_files(mode=mode, folder_num=people_num,
                                                                   file_num=data_per_people, k=1)
    train_dataset = loader.MyDataset(data_dict=train_dict, people_num=people_num, train=True,
                                     mel=mel, noise=noise)
    test_dataset = loader.MyDataset(data_dict=test_dict, people_num=people_num, train=False,
                                    mel=mel, noise=noise)
    print(len(train_dataset), len(test_dataset))
    train_ = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=num_works, pin_memory=True,
                        persistent_workers=persistent, prefetch_factor=prefetch_factor)
    test_ = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=num_works, pin_memory=True,
                       persistent_workers=persistent, prefetch_factor=prefetch_factor)

    # pth_path = 'test.pth'
    # model2 = load_model(pth_path, people_num, Device, not_grad=True)

    model2 = Pretrain_TDNN(people_num, 1024, output_embedding=False, not_grad=False)
    model2.load_parameters('param.model', Device)

    loss = AAMSoftmax(192, people_num, margin, scale, easy_margin)
    writer = SummaryWriter('./logs')
    t.train(train_, test_, model2, loss, Device, writer, num_epochs, learn_rate, weight_decay)
    model2.save_parameters('param2.model')
