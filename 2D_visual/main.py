""""
Example to visualize the decision boundary of 2d points.

Use cases:
# demonet_2d_sum will provide a training acc around 85%
python main.py --model demonet_2d_sum

# demonet_2d_mul will provide a training acc around 95%
python main.py --model demonet_2d_mul

After training, we will save the decision boundary figure to :
    f'{args.model}-acc{str(best_acc)}-Dnoise{args.dataset_noise}-Dsamples{args.dataset_samples}.png'


Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import argparse
import os
import datetime
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from timm.models import create_model
from demonet_2d import *
from timm.loss import LabelSmoothingCrossEntropy
from create_dataset import MoonDataset, ValDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_classes', type=int, default=2, help='batch size in training')
    parser.add_argument('--model', default='demo2net_d4_w100_gelu_sum', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=30, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--has_cuda', action='store_true', default=False)
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--dataset_noise', default=0.4, type=float)
    parser.add_argument('--dataset_samples', default=100, type=int)
    parser.add_argument('--dataset_random_state', default=0, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    has_cuda = torch.cuda.is_available()
    args.has_cuda = has_cuda
    device = 'cuda' if args.has_cuda else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Model
    print(f"args: {args}")
    print('==> Building model..')
    model = create_model(model_name=args.model, num_classes=args.num_classes)
    if args.has_cuda:
        model.cuda()



    print('==> Preparing data, optimizer, scheduler, loss..')
    train_dataset = MoonDataset(n_samples=args.dataset_samples, noise=args.dataset_noise, random_state = args.dataset_random_state )
    train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_val_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.dataset_samples, shuffle=False,
                              drop_last=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    criterion = LabelSmoothingCrossEntropy()
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)

    print('==> Start Training..')
    best_acc = 0
    for epoch in range(0, args.epoch):
        train_out = train(model, train_loader, train_val_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        scheduler.step()
        print(f"==> Epoch: {epoch}, \tLoss: {train_out['loss']}, \tAcc: {train_out['acc']}, \tTime: {train_out['time']}")
        if train_out['acc']> best_acc:
            best_acc = train_out['acc']
            best_model = model

    val_loader = DataLoader(ValDataset(), num_workers=args.workers, batch_size=args.batch_size, shuffle=False,
                              drop_last=False)
    val_out = val(best_model, val_loader,device)
    val_data = val_out["data"]
    val_preds = val_out["preds"]
    color0 = (0.9999, 0.91, 0.99999)
    color1 = (0.8,0.9999,0.8)
    color2 = (0 / 255.0, 204 / 255.0, 204 / 255.0)
    color_list = []
    for i in val_preds:
        if i ==0:
            color_list.append(color0)
        elif i==1:
            color_list.append(color1)
        else:
            color_list.append(color2)
    plt.scatter(val_data[:, 0], val_data[:, 1], label='Scatter Plot', c=color_list, marker='s', s=7)
    train_data =train_dataset.data
    train_labels = train_dataset.labels
    colors = ["orange", "green","pink"]
    train_colors = []
    for i in train_labels:
        train_colors.append(colors[i])
    plt.scatter(train_data[:, 0], train_data[:, 1], label='Scatter Plot', c=train_colors, marker='o')
    # Remove axis lines and ticks
    # plt.subplots_adjust(left=0, right=0, top=0, bottom=0)
    plt.axis('off')

    # Make the plot area tight
    plt.margins(0, 0)
    plt.savefig(f'{args.model}-acc{str(best_acc)}-Dnoise{args.dataset_noise}-Dsamples{args.dataset_samples}.png')
    plt.close()



def train(net, trainloader,train_val_loader, optimizer, criterion, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        total += label.size(0)
        correct += preds.eq(label).sum().item()

    net.eval()
    correct, total = 0, 0
    for batch_idx, (data, label) in enumerate(train_val_loader):
        data, label = data.to(device), label.to(device).squeeze()
        logits = net(data)
        preds = logits.max(dim=1)[1]
        total += label.size(0)
        correct += preds.eq(label).sum().item()

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * correct / total)),
        "time": time_cost
    }



def val(net, valloader, device):
    net.eval()
    preds_list = []
    data_list = []
    for batch_idx, (data) in enumerate(valloader):
        data = data.to(device)
        logits = net(data)
        preds = logits.max(dim=1)[1]  # for different colors, +3
        data_list.append(data)
        preds_list.append(preds)
    print(preds)
    data = torch.cat(data_list, dim=0)
    preds = torch.cat(preds_list, dim=0)
    return {
        "data": data.detach().cpu().numpy(),
        "preds":preds.detach().cpu().numpy()
    }

if __name__ == '__main__':
    main()
