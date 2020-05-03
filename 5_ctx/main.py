import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from load_data import load_tensor_data
from train import *

import time

from model.resnet18 import ResNet18 as Net
# from model.res2net import res2net50 as Net

# 初始化
num_classes = 10  # 汉字类别数量 <=600
num_sample = num_classes*200  # 训练样本数量
batch_size = 64  # batch_size
max_epoch = 13  # epoch
net_name = 'ResNet18'  # 模型类别
load_model = False  # 是否读取模型
save_model = True  # 是否保存模型
is_show_misclassified = True  # 是否展示分类错误图片
loss_func='' #损失函数,不指定时为默认
focal_gamma=2 # 


train_dataset, test_dataset = load_tensor_data(num_classes)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# 模型
this_model = ''
PATH = './model_cache/' + net_name + '_' + this_model + '.pt'  # 模型保存路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(num_classes=num_classes)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-3)

# 读取模型缓存
curr_epoch = 1
if(load_model):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['epoch'] + 1

train_acc_list = []
test_acc_list = []
misclassified_list = []  # 分类错误的图像

start_time = time.time()  # 开始时间
for epoch in range(curr_epoch, max_epoch):
    adjust_learning_rate(optimizer, epoch)

    train_acc = train(train_loader, epoch, model, optimizer,
                      device, loss_func=loss_func, focal_gamma=focal_gamma)  # train
    test_acc, misclassified_list = test(
        train_loader, test_loader, model, device, loss_func=loss_func, focal_gamma=focal_gamma)  # test

    # 保存模型
    if(save_model):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, PATH)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
end_time = time.time()  # 结束时间
train_time = format_time(int(end_time - start_time))
print("训练时间",train_time)

#展示分类错误图片
if(is_show_misclassified):
    show_misclassified_img(misclassified_list)

plot_acc(range(curr_epoch, max_epoch), train_acc_list, test_acc_list, net=net_name,
         num_sample=num_sample, num_classes=num_classes, train_time=train_time)
