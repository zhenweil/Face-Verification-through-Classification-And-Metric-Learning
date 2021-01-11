import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import csv
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, criterion, optimizer, epoch):
    iteration = 1
    total_iteration = len(train_loader)
    model.train()
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        anchor_feat = model.features(anchor)
        positive_feat = model.features(positive)
        negative_feat = model.features(negative)
        loss = criterion(anchor_feat, positive_feat, negative_feat)
        loss.backward()
        optimizer.step()
        if(iteration % 30 == 0):
            print("epoch: {}, iteration: {}/{}, loss: {}".format(epoch, batch_idx, total_iteration, loss.item()))
        iteration += 1
        torch.cuda.empty_cache()

def validation(model, dev_loader, threshold):
    model.eval()
    iteration = 1
    total_iteration = len(dev_loader)
    similarity_result = []
    label_result = []
    with torch.no_grad():
        for batch_idx, (image1, image2, label) in enumerate(dev_loader):
            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)
            feat1 = model.features(image1)
            feat2 = model.features(image2)
            feat1 = feat1.view(feat1.shape[0],-1).detach()
            feat2 = feat2.view(feat2.shape[0],-1).detach()
            similarity = F.cosine_similarity(feat1, feat2)
            similarity = similarity.to('cpu').numpy()
            similarity_result += list(similarity)
            label = label.numpy()
            label_result += list(label)
            if(iteration % 50 == 0):
                auc = roc_auc_score(label_result, similarity_result)
                print("iteration: {}/{}, auc: {}".format(batch_idx, total_iteration, auc))
            iteration += 1
    return auc

def test(model, test_loader):
    model.eval()
    iteration = 1
    total_iteration = len(test_loader)
    similarity_result = []
    label_result = []
    with torch.no_grad():
        for batch_idx, (image1, image2) in enumerate(test_loader):
            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)
            feat1 = model.features(image1)
            feat2 = model.features(image2)
            feat1 = feat1.view(feat1.shape[0],-1).detach()
            feat2 = feat2.view(feat2.shape[0],-1).detach()
            similarity = F.cosine_similarity(feat1, feat2)
            similarity = similarity.to('cpu').numpy()
            similarity_result += list(similarity)
            if(iteration % 50 == 0):
                print("iteration: {}/{}".format(batch_idx, total_iteration))
            iteration += 1
    files = open('../verification_pairs_test.txt').read().splitlines()
    num_pairs = len(files)
    rows = []
    for i in range(num_pairs):
        rows.append([files[i], similarity_result[i]])
    fields = ['Id','Category']
    filename = "../result/prediction.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
    print("finished")