import torch.nn as nn
import torch
class Cnn_Lstm(nn.Module):
    def __init__(self,in_channel,n_classess):
        super(Cnn_Lstm,self).__init__()
        self.in_channel = in_channel
        self.n_classess = n_classess
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512*8*8,1000)
        )
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=1000, hidden_size=256, num_layers=2, bidirectional=True,batch_first=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512,self.n_classess),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
    def forward(self, x):
        feature = []
        for i in range(x.size(1)):
            temp = self.conv1(x[:,i].unsqueeze(1))
            temp = self.conv2(temp)
            temp = self.conv3(temp)
            temp = self.conv4(temp)
            temp = self.fc1(temp.view(temp.size(0),-1))
            feature.append(temp)
        feature = torch.stack(feature,dim=1)
        feature,_ = self.lstm(feature)
        feature = self.fc2(feature[:,-1,:])
        return feature