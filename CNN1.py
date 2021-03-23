import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

path = "./CNN_data/Laser_Data"
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),
                                     transform = transform)
              for x in ["train", "val"]}

data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                batch_size = 20,
                                                shuffle = True)
                     for x in ["train", "val"]}
use_gpu = torch.cuda.is_available()
print(use_gpu)

classes = data_image["train"].classes
classes_index = data_image["train"].class_to_idx

X_train, y_train = next(iter(data_loader_image["train"]))
mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1,2,0))
img = img*std+mean

model = models.vgg16(pretrained=True)

for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 3))

if use_gpu:
    model = model.cuda()

cost = torch.nn.CrossEntropyLoss()
#cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

n_epochs = 100

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
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            y_pred = model(X)
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
            if batch % 20 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct // (4 * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct // len(data_image[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))
