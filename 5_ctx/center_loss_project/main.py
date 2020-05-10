import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import pickle
from six.moves import cPickle

import models
from utils import AverageMeter, Logger, adjust_learning_rate
from center_loss import CenterLoss

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--lr-model', type=float, default=1e-2, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
# misc
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')

###------初始化--------------
num_classes = 200
batch_size  = 128
center_weight = 1
model_name = 'resnet18' #resnet18
max_epoch = 15 #11
feature_dim = 64
eval_freq = 1
loader_path = './temp/c200_n400_s28.pkl'

if_plot = False
if(feature_dim == 2):
    if_plot = True


parser.add_argument('--batch-size', type=int, default=batch_size)            #1. batch size                 
parser.add_argument('--weight-cent', type=float, default=center_weight,       #2. center loss weight
help="weight for center loss")

parser.add_argument('--model', type=str, default=model_name)             #3. model name
parser.add_argument('--max-epoch', type=int, default=max_epoch)              #4. max epoch
parser.add_argument('--featdim', type=int, default=feature_dim)               #5. num_features
parser.add_argument('--eval-freq', type=int, default=eval_freq)               #6. eval frequency                                                   #7. num of classes
parser.add_argument('--plot', action='store_true',default=if_plot)     #8. plot result
args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")


    with open(loader_path, 'rb') as f:
        trainloader, testloader = pickle.load(f)


    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=num_classes, feature_dim=feature_dim)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=args.featdim, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    total_loss_list = []
    train_acc, test_acc = 0, 0
    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer_model, epoch)

        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        loss_list, train_acc = train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              trainloader, use_gpu, num_classes, epoch)
        total_loss_list.append(loss_list)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            test_acc = test(model, testloader, use_gpu, num_classes, epoch)

    total_loss_list = np.array(total_loss_list).ravel()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return total_loss_list,train_acc, test_acc

def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()
    
    correct, total,total_loss = 0, 0, 0
    loss_list = []

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        features, outputs = model(data)
        #准确率
        predictions = outputs.data.max(1)[1]
        total += labels.size(0)
        correct += (predictions == labels.data).sum().item()

        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        total_loss += loss.item()
        loss_list.append(loss.item())

        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')

    train_acc = correct / total
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})'.format(
        total_loss, correct, total,
        train_acc))
    return loss_list,train_acc

def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0

    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum().item()
            
            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    test_acc = correct / total
   
    print('Test set: Accuracy: {}/{} ({:.4f})\n\n'.format(
        correct, total,
        test_acc))
    return test_acc

def plot_features(features, labels, num_classes, epoch, prefix):
    print('-----------------plot features-----------------')
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9','C10','C11','C12','C13','C14']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def plot_loss(loss_list):
    for losses in loss_list:
        plt.plot(losses)
    plt.legend(['weight_cent=0.001','weight_cent=0.01','weight_cent=0.1','weight_cent=1'], loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    weight_cents = [0.001, 0.01, 0.1]
    # weight_cents = [1]
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for weight_cent in weight_cents:
        args.weight_cent = weight_cent
        losses,train_acc,test_acc = main()
        loss_list.append(losses)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)


    with open('./log/data.pkl', 'wb') as f:
        cPickle.dump((loss_list,train_acc_list,test_acc_list), f)
    plot_loss(loss_list)
    print('train_acc',train_acc_list)
    print('test_acc',test_acc_list)






