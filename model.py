import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import time
import random

random.seed(811)
np.random.seed(811)
torch.manual_seed(811)
torch.cuda.manual_seed(811)
torch.cuda.manual_seed_all(811)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_dir = './'

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])


batch_size = 16
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, resnet):
        super(Classifier, self).__init__()
        self.resnet = resnet
        self.linear_network = nn.Sequential(
            nn.Linear(1000, 512),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(512, 7),
            nn.Sigmoid(),
        ) 

    def forward(self, x):
        x = self.resnet(x)
        x = self.linear_network(x)
        return x

resnet = models.resnet18(pretrained=True).cuda()
model = Classifier(resnet = resnet).cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-6)
num_epoch = 100

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    if(epoch == 99 or epoch == 0):
        true_label = []
        predict_label = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

            if(epoch == 99 or epoch == 0):
                for i in np.argmax(val_pred.cpu().data.numpy(), axis=1):
                    predict_label.append(i)
                for i in data[1].numpy():
                    true_label.append(i)

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    if(epoch == 99 or epoch == 0):
        predict_label = np.array(predict_label)
        true_label = np.array(true_label)
        np.save('predict_label.npy', predict_label)
        np.save('true_label.npy', true_label)
        torch.save(model, "{}/model_noval.model".format(model_dir))