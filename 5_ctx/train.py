
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from torch.nn import init

# def train(train_loader,epoch,model,optimizer,device):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
        
#         loss = F.nll_loss(output, target)

#         if batch_idx % 20 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

#         optimizer.zero_grad()   # 所有参数的梯度清零
#         loss.backward()         #即反向传播求梯度
#         optimizer.step()        #调用optimizer进行梯度下降更新参数
def train(train_loader,epoch,model,optimizer,device):
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = F.nll_loss(output, target)

        #叠加损失
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})'.format(
        train_loss, train_correct, len(train_loader.dataset),
        train_acc))
    
    return train_acc

def test(train_loader,test_loader,model,device):
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        test_acc))
    
    return test_acc

def plot_acc(x, train_acc_arr, test_acc_arr, net, num_sample, num_classes):
    plt.figure(figsize=(12,8), dpi=80)  # 图片长宽和清晰度
    plt.grid(True, linestyle="--", alpha=0.5)   # 网格

    train_acc = train_acc_arr[-1]
    test_acc = test_acc_arr[-1]

    plt.title('net:%s,  num_sample:%d,  num_classes:%d,  train_acc:%.3f,  test_acc:%.3f'%(net,num_sample,
    num_classes,train_acc,test_acc))

    plt.plot(x,train_acc_arr, label='train_acc')
    plt.plot(x,test_acc_arr, label='test_acc')
    plt.legend(loc = 'lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


#权重初始化
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()