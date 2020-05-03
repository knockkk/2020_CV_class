import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from torch.nn import init
import matplotlib.pyplot as plt 
import pickle

from focal_loss import focal_loss

def train(train_loader, epoch, model, optimizer, device, loss_func, focal_gamma=0.2):
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        if(loss_func == 'focal_loss'):
            loss = focal_loss(focal_gamma, output, target)
        else:
            criterion = nn.CrossEntropyLoss().cuda()
            loss = criterion(output, target)

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

def test(train_loader,test_loader,model,device,loss_func, focal_gamma):
    test_loss = 0
    test_correct = 0
    misclassified_list = [] #分类错误的图片[a, b]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss
            if(loss_func == 'focal_loss'):
                loss = focal_loss(focal_gamma, output, target)
            else:
                criterion = nn.CrossEntropyLoss().cuda()
                loss = criterion(output, target)
            test_loss += loss.item()

            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            
            test_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            misclassified_list.append([data,pred,target])
            
    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n\n'.format(
        test_loss, test_correct, len(test_loader.dataset),
        test_acc))
    
    return test_acc, misclassified_list

def plot_acc(x, train_acc_arr, test_acc_arr, net, num_sample, num_classes, train_time):
    plt.figure(figsize=(12,8), dpi=80)  # 图片长宽和清晰度
    plt.grid(True, linestyle="--", alpha=0.5)   # 网格

    train_acc = train_acc_arr[-1]
    test_acc = test_acc_arr[-1]

    plt.title('Net:%s,  num_sample:%d,  num_classes:%d,  train_acc:%.3f,  test_acc:%.3f,  train_time:%s'%(net,num_sample,
    num_classes,train_acc,test_acc,train_time))

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

#格式化训练时间
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{0}h {1:02d}min {2:02d}s".format(h, m, s)


#处理分类错误的图像
def show_misclassified_img(misclassified_list):
    new_misclassfied_list = []
    for batch in misclassified_list:
        data = batch[0]
        pred = batch[1]
        target = batch[2]

        pred = pred.cpu().numpy().ravel()
        target = target.cpu().numpy().ravel()
        data = data.cpu().numpy().astype('int32').transpose(0,2,3,1)
        
        misclassified_index = (pred != target)
        if(len(misclassified_index) > 0):
            data = data[misclassified_index] #[]
            pred = pred[misclassified_index] #[]
            target = target[misclassified_index] #[]
            
            for i in range(len(data)):
                new_misclassfied_list.append([data[i],pred[i],target[i]])
    #画图
    draw_imgs(new_misclassfied_list)

def draw_imgs(img_list):
    with open('./temp/char_600.pkl', 'rb') as f:
        char_600 = pickle.load(f)
    row = 8
    column = 8

    plt.figure(1, figsize=(12,12))
    for i in range(1, column * row +1):
        if(i>len(img_list)):
            return
        plt.subplot(row, column, i)
        img = img_list[i-1][0]
        pred = char_600[img_list[i-1][1]]
        target = char_600[img_list[i-1][2]]
        plt.imshow(img)
        plt.text(5, 23, pred, family='SimHei', fontsize=15, fontweight='bold', color='red')
        plt.text(13, 23, target, family='SimHei', fontsize=15, fontweight='bold', color='blue')
        plt.axis('off') # 去掉每个子图的坐标轴
    plt.show()

    plt.figure(2, figsize=(12,12))
    for i in range(1, column * row +1):
        i = i + column * row 
        if(i > len(img_list)):
            return
        plt.subplot(row, column, i - column * row)
        img = img_list[i-1][0]
        pred = char_600[img_list[i-1][1]]
        target = char_600[img_list[i-1][2]]
        plt.imshow(img)
        plt.text(5, 23, pred, family='SimHei', fontsize=15, fontweight='bold', color='red')
        plt.text(13, 23, target, family='SimHei', fontsize=15, fontweight='bold', color='blue')
        plt.axis('off') # 去掉每个子图的坐标轴
    plt.show()

#动态调整学习率
def adjust_learning_rate(optimizer, epoch):
    lr = 0.01
    if(epoch>=7):
        lr = 2e-3
    if(epoch>=10):
        lr = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("current lr", lr)

