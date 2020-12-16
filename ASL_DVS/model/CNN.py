import torch
class CNNnet(torch.nn.Module):
    def __init__(self,in_channel,n_classess):
        super(CNNnet,self).__init__()
        self.in_channel = in_channel
        self.n_classess = n_classess
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channel,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128,256,3,1,1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256,256,3,1,1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256,512,3,1,1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512,512,3,1,1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(8*8*512,1000),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1000, self.n_classess),
            torch.nn.Dropout(0.5),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x.view(x.size(0),-1))
        return x