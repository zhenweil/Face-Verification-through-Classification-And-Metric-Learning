import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torch.optim import Adam, SGD, lr_scheduler
from dataloader import train_dataset, dev_dataset
from torch.utils.data import DataLoader
from torch import nn
from train_test import train, validation, test
import sys
sys.path.append("../model")
from models import ResNet18, TripletLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
param = {
    'resume': False,
    'resume_from': -1,
    'load checkpoint from N_way_classification': False,
    'checkPointPath': '../checkpoints/metric_learning',
    'lr': 0.0001,
    'nepochs': 10,
    'img_per_person': 10,
    'thresh': 1,
    'train_batch_size': 64,
    'dev_batch_size': 100
    }

def main():
    '''model = models.resnet18(pretrained = False)
    model.fc = nn.Linear(512, 4000)'''
    model = ResNet18(4000)
    if(param['resume'] == True):
        if(param['load checkpoint from N_way_classification'] == True):
            checkPointPath = '../checkpoints/N_way_classification' + '/epoch' + str(param['resume_from'])
            checkpoint = torch.load(checkPointPath)
            model.load_state_dict(checkpoint['model_state_dict'])
            '''modules = list(model.children())[:-1]
            model = nn.Sequential(*modules)'''
        else:
            checkPointPath = param['checkPointPath'] + '/epoch' + str(param['resume_from'])
            checkpoint = torch.load(checkPointPath)
            model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr = 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.95)
    criterion = TripletLoss(margin = 1, img_per_person = param['img_per_person'])
    criterion.to(DEVICE)

    data_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_root = "../classification_data/train_data"
    dev_path = '../verification_pairs_val.txt'
    test_path = '../verification_pairs_test.txt'

    train_data = train_dataset(train_root, param['img_per_person'], data_transform)
    dev_data = dev_dataset(dev_path, data_transform)
    test_data = dev_dataset(test_path, data_transform)

    train_loader = DataLoader(train_data, batch_size = param['train_batch_size'], pin_memory = True, shuffle = True)
    dev_loader = DataLoader(dev_data, batch_size = param['dev_batch_size'], pin_memory = True, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = param['dev_batch_size'], pin_memory = True, shuffle = False)

    print("start training")
    start_epoch = param['resume_from'] + 1
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, start_epoch + param['nepochs']):
        train(model, train_loader, criterion, optimizer, epoch)
        path = param['checkPointPath'] + "/epoch" + str(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        scheduler.step()
        auc = validation(model, dev_loader, param['thresh'])
        print("validation auc is: ", auc)
    test(model, test_loader)

if __name__ == '__main__':
    main()