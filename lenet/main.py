import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time
from PIL import Image
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())
def load_data_fashion_mnist(mnist_train,mnist_test,batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
        train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        return train_iter,test_iter
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(mnist_train,mnist_test, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LeNet (nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output
net=LeNet()
print(net)
def evaluate_accuracy(data_iter,net,device=None):
    if device is None and isinstance(net,torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum,n = 0.0, 0
    with torch.no_grad():
        for X,y in data_iter:
            net.eval()
            acc_sum +=(net(X.to(device)).argmax(dim=1) ==y.to(device)).float().sum().cpu().item()
            net.train()
            n +=y.shape[0]
    return acc_sum / n
def train(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net=net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum, n ,batch_count, start =0.0, 0.0, 0, 0,time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'
           % (epoch +1 ,train_l_sum/batch_count,train_acc_sum / n, test_acc,time.time()-start))
lr,num_epochs = 0.0001,30
optimizer=torch.optim.Adam(net.parameters(),lr=lr)
train(net, train_iter, test_iter,batch_size,optimizer,device,num_epochs)
print('Finish Training')
save_path='./Lenet.pth'
torch.save(net.state_dict(), save_path)
print("Testing")
lenet = LeNet()
lenet.load_state_dict(torch.load("Lenet.pth"))
test_images = Image.open('8.png')
img_to_tensor=transforms.Compose([
    transforms.Resize([28,28]),

    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
input_images=img_to_tensor(test_images).unsqueeze(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    input_images = input_images.cuda()
    lenet =lenet.cuda()
output_data = lenet(input_images)
test_labels = torch.max(output_data,1)[1].data.cpu().numpy().squeeze(0)
print(output_data)
print(test_labels)