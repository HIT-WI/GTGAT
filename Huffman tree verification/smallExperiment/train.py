#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy,makedata
from model import LinearModel,RandomTree,HufTree
import time

chose_features,chose_labels,chose_weight_dict,idx_train,idx_val,idx_test = makedata()

l_model = LinearModel(chose_features.shape[1],7)
r_model = RandomTree(chose_features.shape[1],8,7,chose_weight_dict,0.2)
h_model = HufTree(chose_features.shape[1],8,7,chose_weight_dict,0.2)

optimer1 = optim.Adam(l_model.parameters(), lr=0.01, weight_decay=1e-5)
optimer2 = optim.Adam(r_model.parameters(), lr=0.01, weight_decay=1e-5)
optimer3 = optim.Adam(h_model.parameters(), lr=0.01, weight_decay=1e-5)

def train(epoch):
    t = time.time()
    optimer1.zero_grad()
    optimer2.zero_grad()
    optimer3.zero_grad()
    output1 = l_model(chose_features)
    output2 = r_model(chose_features)
    output3 = h_model(chose_features)
    loss_train1 = F.nll_loss(output1[idx_train], chose_labels[idx_train])
    acc_train1 = accuracy(output1[idx_train], chose_labels[idx_train])
    loss_train2 = F.nll_loss(output2[idx_train], chose_labels[idx_train])
    acc_train2 = accuracy(output2[idx_train], chose_labels[idx_train])
    loss_train3 = F.nll_loss(output3[idx_train], chose_labels[idx_train])
    acc_train3 = accuracy(output3[idx_train], chose_labels[idx_train])


    loss_val1 = F.nll_loss(output1[idx_val], chose_labels[idx_val])
    acc_val1 = accuracy(output1[idx_val], chose_labels[idx_val])
    loss_val2 = F.nll_loss(output2[idx_val], chose_labels[idx_val])
    acc_val2 = accuracy(output2[idx_val], chose_labels[idx_val])
    loss_val3 = F.nll_loss(output3[idx_val], chose_labels[idx_val])
    acc_val3 = accuracy(output3[idx_val], chose_labels[idx_val])
    loss_train1.backward()
    loss_train2.backward()
    loss_train3.backward()
    optimer1.step()
    optimer2.step()
    optimer3.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train1: {:.4f}'.format(loss_train1.data.item()),
          'acc_train1: {:.4f}'.format(acc_train1.data.item()),
          'loss_val1: {:.4f}'.format(loss_val1.data.item()),
          'acc_val1: {:.4f}'.format(acc_val1.data.item()))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train2: {:.4f}'.format(loss_train2.data.item()),
          'acc_train2: {:.4f}'.format(acc_train2.data.item()),
          'loss_val2: {:.4f}'.format(loss_val2.data.item()),
          'acc_val2: {:.4f}'.format(acc_val2.data.item()))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train3: {:.4f}'.format(loss_train3.data.item()),
          'acc_train3: {:.4f}'.format(acc_train3.data.item()),
          'loss_val3: {:.4f}'.format(loss_val3.data.item()),
          'acc_val3: {:.4f}'.format(acc_val3.data.item()))
    return loss_val1.data.item(),loss_val2.data.item(),loss_val3.data.item()

def compute_test():
    output1 = l_model(chose_features)
    output2 = r_model(chose_features)
    output3 = h_model(chose_features)

    loss_test1 = F.nll_loss(output1[idx_test], chose_labels[idx_test])
    acc_test1 = accuracy(output1[idx_test], chose_labels[idx_test])
    loss_test2 = F.nll_loss(output2[idx_test], chose_labels[idx_test])
    acc_test2 = accuracy(output2[idx_test], chose_labels[idx_test])
    loss_test3 = F.nll_loss(output3[idx_test], chose_labels[idx_test])
    acc_test3 = accuracy(output3[idx_test], chose_labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test1.data.item()),
          "accuracy= {:.4f}".format(acc_test1.data.item()))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test2.data.item()),
          "accuracy= {:.4f}".format(acc_test2.data.item()))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test3.data.item()),
          "accuracy= {:.4f}".format(acc_test2.data.item()))
    return acc_test1.data.item(),acc_test2.data.item(),acc_test3.data.item()


if __name__ == "__main__":
    loss_vals1 = []
    loss_vals2 = []
    loss_vals3 = []
    best1 = 10000
    best_epoch1 = 0
    best2 = 10000
    best_epoch2 = 0
    best3 = 10000
    best_epoch3 = 0
    for epoch in range(300):
        loss1,loss2,loss3 = train(epoch)
        loss_vals1.append(loss1)
        loss_vals2.append(loss2)
        loss_vals3.append(loss3)
        if loss_vals1[-1] < best1:
            best1 = loss_vals1[-1]
            best_epoch1 = epoch
        if loss_vals2[-1] < best2:
            best2 = loss_vals2[-1]
            best_epoch2 = epoch
        if loss_vals3[-1] < best3:
            best3 = loss_vals3[-1]
            best_epoch3 = epoch

