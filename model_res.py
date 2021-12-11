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
import pandas as pd
from resnet18 import ResNet18

random.seed(811)
np.random.seed(811)
torch.manual_seed(811)
torch.cuda.manual_seed(811)
torch.cuda.manual_seed_all(811)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_img(df, img_type):
    type_len = 0
    for i in df:
        if(i[2] == img_type):
            type_len += 1
    x = np.zeros((type_len, 48, 48, 3), dtype=np.uint8)
    y = np.zeros((type_len), dtype=np.uint8)     
    temp = 0  
    for i in df:
        if(i[2] == img_type):
            arr = i[1].split()
            for k, j in enumerate(arr):
                arr[k] = int(j)
            arr = np.array(arr).reshape((48, 48))
            arr = cv2.cvtColor(np.float32(arr), cv2.COLOR_GRAY2BGR)
            x[temp, :, :, :] = arr
            y[temp] = i[0]
            temp += 1
    return x, y



df = pd.read_csv('./fer2013.csv').to_numpy()
print("Reading data")
train_x, train_y = read_img(df, 'Training')
val_x, val_y = read_img(df, 'PublicTest') 

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

class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        # label is required to be a LongTensor
        self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        Y = self.y[index]
        return X, Y


batch_size = 128
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

ori_lr = 0.001
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

#resnet = models.resnet34(pretrained=True).cuda()
#model = Classifier(resnet = resnet).cuda()
model = ResNet18().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=ori_lr, weight_decay=1e-6)
num_epoch = 100

print("Starting training")
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()

    if epoch > 50:
        optimizer = torch.optim.SGD(model.parameters(), lr=ori_lr, weight_decay=5e-4)

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = ori_lr * decay_factor
        set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = ori_lr
    print('learning_rate: %s' % str(current_lr))

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    #if(epoch == num_epoch - 1):
    #    true_label = []
    #    predict_label = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

            '''
            if(epoch == num_epoch - 1):
                for i in np.argmax(val_pred.cpu().data.numpy(), axis=1):
                    predict_label.append(i)
                for i in data[1].numpy():
                    true_label.append(i)
            '''

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    if(epoch == num_epoch - 1):
        #predict_label = np.array(predict_label)
        #true_label = np.array(true_label)
        #np.save('predict_label_res18.npy', predict_label)
        #np.save('true_label_res18.npy', true_label)
        torch.save(model, "{}/model_res18_100_nomask_opt.model".format(model_dir))