import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace = True)
        if(stride != 1 or inchannel != outchannel):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        indentity = self.shortcut(x)
        out = self.relu(out + indentity)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(BasicBlock, 64, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_layer(self, basic_block, outchannel, stride=1):
        strides = [stride, 1]
        layers = []
        for stride in strides:
            layers.append(basic_block(self.inchannel, outchannel, stride))
            self.inchannel = outchannel
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        embedding = out.view(out.size(0),-1)
        return embedding
        
    def forward(self, x):
        embedding = self.features(x)
        out = self.fc(embedding)
        return out

class TripletLoss(nn.Module):
    def __init__(self, margin = 1, img_per_person = 10):
        super(TripletLoss, self).__init__()
        self.margin = torch.tensor(margin)
        self.img_per_person = img_per_person
        self.relu = nn.ReLU()
    def forward(self, anchor, positive, negative):
        d_ap = (anchor - positive).pow(2).sum(1).pow(0.5)
        d_an = (anchor - negative).pow(2).sum(1).pow(0.5)
        loss = self.relu(d_ap - d_an + self.margin)
        return loss.mean()
                    
