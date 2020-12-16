#添加dropout
import torchvision
from tqdm import tqdm
import torch
from torch import nn
from model.CNN import CNNnet
from model.Resnet import resnet18,resnet34
from model.CL import Cnn_Lstm
from model.Base_model import Cnn
import os
import numpy as np
from PIL import Image
from torch.autograd import Variable
from deel_data.load_data import load_data
import matplotlib.pyplot as plt

batch_size = 64
path = "/home/xavier/data/DvsGesture_frame_filter"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device :", device)
model = Cnn(16,10)
model.to(device)
'''if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)'''
lr_step = [100,200]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_step,gamma = 0.1)
def train():
    train_acc_list =[]
    val_acc_list = []
    path_list = os.listdir(path)
    path_list.sort(key=lambda x : x)
    train_path_list = path_list[:110]
    val_path_list = path_list[110:]
    train_path_x_y,val_path_x_y = [],[]

    for train_path in train_path_list:
        temp = os.listdir(os.path.join(path, train_path))
        temp.sort(key=lambda x: int(x))
        for label in temp[:-1]:
            counts = os.listdir(os.path.join(path, train_path, label))
            for count in counts:
                train_path_x_y.append([os.path.join(path, train_path, label, count), int(label)])
    train_path_x_y = np.array(train_path_x_y)

    for val_path in val_path_list:
        temp = os.listdir(os.path.join(path, val_path))
        temp.sort(key=lambda x: int(x))
        for label in temp[:-1]:
            counts = os.listdir(os.path.join(path, val_path, label))
            for count in counts:
                val_path_x_y.append([os.path.join(path, val_path, label, count), int(label)])
    val_path_x_y = np.array(val_path_x_y)

    acc_min = 0
    length_train = len(train_path_x_y)
    for epoch in range(400):
        train_loss = []
        acc = []
        model.train()
        np.random.shuffle(train_path_x_y)
        for i in tqdm(range(length_train//batch_size)):
            temp = train_path_x_y[i*batch_size:(i+1)*batch_size]
            x,y = load_data(temp,16)
            x = Variable(x).to(device,dtype=torch.float32)
            y = Variable(y).to(device)
            out = model(x)
            loss = criterion(out,y)
            train_loss.append(loss.item())
            accuracy = torch.max(out.cpu(), 1)[1].numpy() == y.cpu().numpy()
            acc.append(accuracy.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = val(val_path_x_y)
        train_acc_list.append(np.array(acc).mean())
        val_acc_list.append(val_acc)
        if(val_acc>acc_min):
            torch.save(model,"/home/xavier/test_compare/test1")
            acc_min = val_acc
        print("train_acc:",np.array(acc).mean(),"||val_acc:",val_acc)
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.show()

def val(val_path_x_y):
    model.eval()
    val_acc = []
    length_val = len(val_path_x_y)
    with torch.no_grad():
        for i in range(length_val//batch_size):
            temp = val_path_x_y[i*batch_size:(i+1)*batch_size]
            x,y = load_data(temp,16)
            x = Variable(x).to(device,dtype=torch.float32)
            y = Variable(y).to(device)
            out = model(x)
            accuracy = torch.max(out.cpu(), 1)[1].numpy() == y.cpu().numpy()
            val_acc.append(accuracy.mean())
    return np.array(val_acc).mean()
if __name__ == '__main__':
    train()