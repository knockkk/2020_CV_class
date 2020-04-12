import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from load_data import load_tensor_data
from train import *

# from model.simple_net import Net
from model.resnet18 import ResNet18 as Net

#获取数据
num_classes = 600 #汉字类别数量
train_dataset, test_dataset = load_tensor_data(num_classes)

batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(num_classes)

# model.apply(weigth_init) #权重初始化

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
# optimizer = optim.Adadelta(model.parameters(), lr=1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
train_acc_list = []
test_acc_list = []
epoches = 10
for epoch in range(1,epoches):
    train_acc = train(train_loader,epoch,model,optimizer,device) #train 
    test_acc = test(train_loader,test_loader,model,device)  #test
    
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

plot_acc(range(1,epoches),train_acc_list,test_acc_list, net='ResNet18', 
                            num_sample=num_classes*180, num_classes=num_classes)

