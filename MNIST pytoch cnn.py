import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import time  
import warnings


#设置神经网络的超参数
epochs = 5 #迭代次数
num_classes = 10 # 0-9分十类
batch_size = 32 #每一次扔进神经网络中32张图
lr = 0.001 #learning rate


#构建pipl对图像进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(),#将图片转换为tensor格式
    transforms.Normalize((0.1307,),(0.3081,)) #各个通道的均值和标准差，使数据变成均值为0，标准差为1的正态分布
    
])

path = osp.join(os.getcwd(), 'data')#在当前目录下创建data文件夹供之后数据下载储存使用
print(path)  

#训练数据集
train_dataset = torchvision.datasets.MNIST(root = path,
                                          train = True,
                                          transform = pipeline,
                                          download = True)
#测试数据集
test_dataset = torchvision.datasets.MNIST(root = path,
                                          train =False,
                                          transform = pipeline,
                                          download = True)

#加载数据
train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle= True )
test_loader = DataLoader(test_dataset,batch_size = 1000, shuffle= True )


#构建网络模型

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)#1表示颜色是灰色，10表示10个卷积核，5表示卷积核的尺寸是5x5
        self.pool1 = nn.MaxPool2d(2,2)# 2×2的最大池化层
        self.conv2 = nn.Conv2d(10,20,3)#20个卷积核，尺寸是3x3
        self.pool2 = nn.MaxPool2d(2,2)# 2×2的最大池化层
        self.fc1 = nn.Linear(20*5*5,128)#128个神经元的线性层
        self.fc2 = nn.Linear(128,10)#输出层
        
    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x =  nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,20*5*5) #展平准备进入MLP网络
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x,dim=1) #

# 训练模型
def train_test_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_loss = []
    test_accuracy = []
 
    # train training set
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_samples = 0
        for i, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()*X_train.size(0) #每一个batch中的损失加在一起
            num_samples += X_train.size(0)#每一个batch_size中的图片数量加在一起
            
        epoch_loss = epoch_loss / num_samples
        train_loss.append(epoch_loss)
            
            
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy())
            accuracy = skm.accuracy_score(y_true, y_pred)
            test_accuracy.append(accuracy)
            
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
                
#实例化模型
model = CNN()   

#定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)




# 调用 train_test_model 函数，并计时
start_time = time.time()
train_test_model(model, train_loader, test_loader, criterion, optimizer, epochs)
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))
            
        
