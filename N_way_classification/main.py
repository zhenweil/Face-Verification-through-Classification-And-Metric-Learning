import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms, datasets
from torch.optim import SGD, lr_scheduler
from dataloader import MyVerificationDataset
from torch.utils.data import DataLoader
from torch import nn
from train_test import train, validation,verification_dev, verification_test
import sys
sys.path.append("../model")
from models import ResNet18

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
param = {
    'resume': False,
    'resume_from': -1,
    'checkPointPath': '../checkpoints/N_way_classification',
    'lr': 0.005,
    'nepochs': 5
}

def main():
    '''model = models.resnet18(pretrained = False)
    model.fc = nn.Linear(512, 4000)'''
    model = ResNet18(4000)
    model.to(DEVICE)
    optimizer = SGD(model.parameters(), lr = 0.15, momentum = 0.9, weight_decay = 5e-5)

    if(param['resume'] == True):
        print("loading from checkpoint {}".format(param['resume_from']))
        checkPointPath = param['checkPointPath'] + '/epoch' + str(param['resume_from'])
        checkpoint = torch.load(checkPointPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        print("finish loading")
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)

    batch_size = 10 if DEVICE == 'cuda' else 1
    data_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root='../classification_data/train_data/', transform = data_transform)
    val_dataset = datasets.ImageFolder(root = '../classification_data/val_data' , transform = data_transform)
    verfication_dev_dataset = MyVerificationDataset('../verification_pairs_val.txt', data_transform)
    verfication_test_dataset = MyVerificationDataset('../verification_pairs_test.txt', data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    dev_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    verification_dev_loader = DataLoader(verfication_dev_dataset, batch_size = batch_size, shuffle = False)
    verification_test_loader = DataLoader(verfication_test_dataset, batch_size = batch_size, shuffle = False)

    
    start_epoch = param['resume_from'] + 1
    torch.cuda.empty_cache()
    acc = validation(model, dev_loader)
    auc = verification_dev(model, verification_dev_loader)
    print("start training")
    for epoch in range(start_epoch, start_epoch + param['nepochs']):
        train(model, train_loader, criterion, optimizer, epoch)
        path = param['checkPointPath'] + "/epoch" + str(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        scheduler.step()
        acc = validation(model, dev_loader)
        auc = verification_dev(model, verification_dev_loader)
        print("auc is: ", auc)

if __name__ == '__main__':
    main()