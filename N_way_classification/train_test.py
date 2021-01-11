import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import csv
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, criterion, optimizer, epoch):
    iteration = 1
    total_iteration = len(train_loader)
    model.train()
    for batch_idx, (feats, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if(iteration % 200 == 0):
            print("epoch: {}, iteration: {}/{}, loss: {}".format(epoch, batch_idx, total_iteration, loss.item()))
        iteration += 1
        torch.cuda.empty_cache()
        del feats
        del labels

def tensor_onehot_to_label(output):    
    output_np = output.detach().to('cpu').numpy()
    label = np.argmax(output_np, axis = 1)
    return label

def calculate_accuracy(output, label):
    output_label = tensor_onehot_to_label(output).astype(int)
    label = label.numpy().astype(int)
    comparison = np.zeros(label.shape)
    comparison[label == output_label] = 1
    num_correct = np.sum(comparison)
    accuracy = num_correct/label.shape[0]
    return accuracy

def validation(model, dev_loader):
    model.eval()
    iteration = 0
    total_iteration = len(dev_loader)
    average_acc = 0
    for batch_idx, (image, label) in enumerate(dev_loader):
        image = image.to(DEVICE)
        out = model(image)
        acc = calculate_accuracy(out, label)
        average_acc += acc
        if(iteration % 30 == 29):
            print("iteration: {}/{}, accuracy: {}".format(batch_idx, total_iteration, acc))
        iteration += 1
    average_acc = average_acc/iteration
    print("average accuracy is: ", average_acc)
    return average_acc

def verification_dev(model, verification_loader):
    model.eval()
    iteration = 1
    total_iteration = len(verification_loader)
    similarity_result = []
    label_result = []
    with torch.no_grad():
        for batch_idx, (image1, image2, label) in enumerate(verification_loader):
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

def verification_test(model, verification_loader):
    model.eval()
    iteration = 1
    total_iteration = len(verification_loader)
    similarity_result = []
    label_result = []
    with torch.no_grad():
        for batch_idx, (image1, image2) in enumerate(verification_loader):
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