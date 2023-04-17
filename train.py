import os
import torch
import loader
import eval as d2l
# from d2l import torch as d2l
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.tdnn import ECAPA_TDNN
from loss import AAMSoftmax, evaluate_accuracy_gpu


def init_logs(path=".\\logs"):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def train(train_iter, test_iter, net, loss_func, device, write, num_epoch=10, lr=0.1, wd=2e-4):
    net.to(device)
    trainer = torch.optim.Adam(params=(param for param in net.parameters()
                                       if param.requires_grad), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(trainer, base_lr=1e-3, max_lr=0.1, step_size_up=6250,
                                                  mode="triangular2", cycle_momentum=False)
    timer = d2l.Timer()
    sum, img = 0, None
    for epoch in range(num_epoch):
        print(f'\nepoch {epoch + 1}:')
        train_acc = train_l = 0
        metric = d2l.Accumulator(3)
        net.train()
        for i, (x, y) in enumerate(train_iter):
            # if i == 0 and epoch == num_epoch - 1:
            #     img = x.to(device)
            timer.start()
            x, y = x.to(device), y.to(device)
            trainer.zero_grad()
            y_hat = net(x)
            l, prec = loss_func(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * x.shape[0], prec * x.shape[0], x.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            scheduler.step()
        sum += metric[2]
        # test_acc = 0
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'\tloss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        write.add_scalar('loss', train_l, epoch)
        write.add_scalars('acc', {'test_acc': test_acc, 'train_acc': train_acc}, epoch)
    print(f'\n{sum / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    # write.add_graph(net, img)


if __name__ == "__main__":
    people_num, data_per_people = 420, 150
    noise, mel = False, True
    margin, scale, easy_margin = 0.2, 20, False
    not_grad, bidirectional, reverse = False, True, False
    num_epochs, learn_rate, weight_decay = 150, 0.125, 1e-3
    mode, model_name = "train", "dense169"
    hidden_size, num_layers = 64, 2
    model_path = './pretrain.model'

    # Device = torch_directml.device()
    # print(Device)
    # prefetch_factor, batch_size, num_works, persistent = 2, 32, 8, False

    Device = d2l.try_gpu()
    if Device.type == 'cpu':
        prefetch_factor, batch_size, num_works, persistent = 2, 8, 8, False
    elif torch.cuda.is_available():
        prefetch_factor, batch_size, num_works, persistent = 8, 256, 32, True
    else:
        prefetch_factor, batch_size, num_works, persistent = 2, 32, 8, False

    init_logs()
    train_dict, test_dict, people_num = loader.load_files(mode=mode, folder_num=people_num,
                                                          file_num=data_per_people, k=7.5)
    train_dataset = loader.MyDataset(data_dict=train_dict, people_num=people_num, train=True, mel=mel,
                                     noise=noise)
    test_dataset = loader.MyDataset(data_dict=test_dict, people_num=people_num, train=False, mel=mel,
                                    noise=False)
    print(len(train_dataset), len(test_dataset))
    train_ = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=num_works, pin_memory=True,
                        persistent_workers=persistent, prefetch_factor=prefetch_factor)
    test_ = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=num_works, pin_memory=True,
                       persistent_workers=persistent, prefetch_factor=prefetch_factor)
    writer = SummaryWriter('./logs')

    # model1 = cnn.get_net(people_num, model_name, not_grad)
    # model2 = F.CNN_LSTM(model_name, people_num, hidden_size, num_layers, bidirectional, not_grad)
    model2 = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192,
                        output_num=people_num, context=True, embedding=False)

    loss = AAMSoftmax(192, people_num, margin, scale, easy_margin)
    train(train_, test_, model2, loss, Device, writer, num_epochs, learn_rate, weight_decay)
    torch.save(model2.state_dict(), "net.model")
