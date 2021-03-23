import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import models,transforms,datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image


class MultiInput(nn.Module):

    def __init__(self):
        super(MultiInput, self).__init__()
        Laser_Module = list(models.googlenet(pretrained=True).children())[:-1]
        self.model_Laser = nn.Sequential(*Laser_Module)
        #FR_Module = list(models.vgg16(pretrained=True).children())[:-1]
        self.model_FeedRate = nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        #Freq_Module = list(models.vgg16(pretrained=True).children())[:-1]
        self.model_Freq = nn.Sequential(
            torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc = nn.Sequential(nn.Linear(38656,4096),  #vgg16:25088  mymodel:18816 resnet18:512 resnet50:2048
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096,4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096,24))
    def forward (self,laser,feedrate,freq):
        a=self.model_Laser(laser)
        b=self.model_FeedRate(feedrate)
        c=self.model_Freq(freq)
        x=torch.cat((a.view(a.size(0),-1),b.view(b.size(0),-1)),dim=1)
        x=torch.cat((x.view(x.size(0),-1),c.view(c.size(0),-1)),dim=1)
        x=self.fc(x)

        return x
model = MultiInput()
# a=Variable(torch.randn(1,3,224,224))
# x_laser = Variable(torch.randn(1,3,224,224))
# x_feedrate = Variable(torch.randn(1,3,224,224))
# out=model(a,x_laser,x_feedrate)
# print(out.numel())

path = ".\\CNN_data\\0319Train_Data"
path2 = ".\\CNN_data\\0319Laser_param"
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),
                                     transform = transform)
              for x in ["train", "val"]}
data_m = {x:datasets.ImageFolder(root = os.path.join(path2,x),
                                     transform = transform)
            for x in ["Speed","Freq"]}


data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                batch_size = 8,
                                                shuffle = False)
                     for x in ["train", "val"]}
data_loader_m = {x:torch.utils.data.DataLoader(dataset=data_m[x],
                                                batch_size = 8,
                                                shuffle = False)
                     for x in ["Speed","Freq"]}

use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()

cost = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 100
train_cor=[]
train_los=[]
val_cor =[]
val_los =[]

for data in data_loader_m['Speed']:
    x,y = data
    if y[0] == 0:
        data_1m = x
    elif y[0] ==1:
        data_2m=x
    else:
        data_3m =x

for data in data_loader_m['Freq']:
    x,y = data
    if y[0] == 0:
        data_20f = x
    elif y[0] ==1:
        data_40f=x
    else:
        data_60f =x


for data in data_loader_image['train']:
    x,y =data
    # print(x.shape)
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            if y[0] == 0 or y[0] == 1 or y[0] == 2 or y[0] == 3 or y[0] == 4 or y[0] == 5 or y[0] == 6 or y[0] == 7:
                input_speed = data_1m
            elif y[0] == 8 or y[0] == 9 or y[0] == 10 or y[0] == 11 or y[0] == 12 or y[0] == 13 or y[0] == 14:
                input_speed = data_2m
            else:
                input_speed = data_3m

            if y[0] == 0 or y[0] == 2 or y[0] == 5 or y[0] == 8 or y[0] == 10 or y[0] == 13 or y[0] == 16 or y[
                0] == 18 or y[0] == 21:
                input_freq = data_20f
            elif y[0] == 1 or y[0] == 3 or y[0] == 6 or y[0] == 9 or y[0] == 11 or y[0] == 14 or y[0] == 17 or y[
                0] == 19 or y[0] == 22:
                input_freq = data_40f
            else:
                input_freq = data_60f

            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
                input_speed, input_freq = Variable(input_speed.cuda()), Variable(input_freq.cuda())
            else:
                X, y = Variable(X), Variable(y)
                input_speed, input_freq = Variable(input_speed), Variable(input_freq)

            optimizer.zero_grad()
            y_pred = model(X,input_speed,input_freq)
            _, pred = torch.max(y_pred.data, 1)
            # print("y_pred:{}".format(y_pred))
            # print("y:{}".format(y))
            # print ("pred:{}".format(pred))
            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(pred == y.data)
            if batch % 8 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct // (4 * batch)))


        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct // len(data_image[param])
        if param == "train":
            train_cor.append(epoch_correct)
            train_los.append(epoch_loss)
        else:
            val_cor.append(epoch_correct)
            val_los.append(epoch_loss)

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

plt.subplot(211)
plt.plot(train_cor)
plt.plot(val_cor)
plt.subplot(212)
plt.plot(train_los)
plt.plot(val_los)

plt.show()

torch.save(model, './model_save/Multi_Input_resnet101v2.pth')
