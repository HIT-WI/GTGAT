#encoding=utf-8

'''
实验分析：
本代码主要从三个方向来对比模型结果：
1. 线性模型
2. 随机树模型
3. huf树模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import sortNodes


#线性模型
class LinearModel(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim,out_dim)
    def forward(self,features):
        outs = self.linear(features)
        result = F.log_softmax(outs,dim=1)
        return result

#随机树模型
class RandomTree(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class,weight_dict,alpha):
        super(RandomTree, self).__init__()
        self.weight_dict = weight_dict
        self.leakyrule = nn.LeakyReLU(alpha)
        self.C = nn.Parameter(torch.zeros(size=(input_dim,hidden_dim)))
        nn.init.xavier_normal_(self.C.data,gain=1.414)
        self.W = nn.Parameter(torch.zeros(size=(2*hidden_dim,hidden_dim)))
        nn.init.xavier_normal_(self.W.data,gain=1.414)
        self.V = nn.Parameter(torch.zeros(size=(hidden_dim,num_class)))
        nn.init.xavier_normal_(self.V.data,gain=1.414)
        self.num_class = num_class
        self.hidden_dim = hidden_dim
    def forward(self,features):
        global neighbor_list, temp_out, x_1, x_2
        h = torch.mm(features,self.C)
        N = features.shape[0]
        outs = torch.zeros(size=(N,self.hidden_dim))
        for idx,neighbor_dict in self.weight_dict.items():
            if len(neighbor_dict) == 1:
                neighbor_list = list(neighbor_dict.keys())
                neighbor_list.append(neighbor_list[0])
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = h[node1_idx,:].reshape(1,-1)
                x_2 = h[node2_idx,:].reshape(1,-1)
                x = torch.cat([x_1,x_2],dim=1)
                temp_out = torch.mm(x,self.W)
            elif len(neighbor_dict) == 2:
                neighbor_list = list(neighbor_dict.keys())
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = h[node1_idx, :].reshape(1, -1)
                x_2 = h[node2_idx, :].reshape(1, -1)
                x = torch.cat([x_1, x_2], dim=1)
                temp_out = torch.mm(x, self.W)
            else:
                neighbor_list = list(neighbor_dict.keys())
                #随机打乱
                #np.random.shuffle(neighbor_list)
                index = 10000
                temp_dict = {}

                while len(neighbor_list) < 2:
                    node1_idx = neighbor_list.pop()
                    node2_idx = neighbor_list.pop()
                    if node1_idx < 10000:
                        x_1 = h[node1_idx, :].reshape(1, -1)
                    else:
                        x_1 = temp_dict[node1_idx].reshape(1,-1)
                    if node2_idx < 10000:
                        x_2 = h[node2_idx,:].reshape(1,-1)
                    else:
                        x_2 = temp_dict[node2_idx].reshape(1,-1)
                    x = torch.cat([x_1, x_2], dim=1)
                    temp_out = torch.mm(x, self.W)
                    neighbor_list.append(index)
                    temp_dict[index] = temp_out
                    index += 1
                node1_idx = neighbor_list.pop()
                if node1_idx < 10000:
                    temp_out = h[node1_idx,:].reshape(1,-1)
                else:
                    temp_out = temp_dict[node1_idx].reshape(1,-1)
                #x = torch.cat([temp_out,x_1],dim=1)
                #temp_out = torch.mm(x,self.W)

            outs[idx] = temp_out
        results = self.leakyrule(torch.mm(outs,self.V))

        return F.log_softmax(results,dim=1)


#huf树模型
class HufTree(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class,weight_dict,alpha):
        super(HufTree,self).__init__()
        self.weight_dict = weight_dict
        self.leakyrule = nn.LeakyReLU(alpha)
        self.C = nn.Parameter(torch.zeros(size=(input_dim, hidden_dim)))
        nn.init.xavier_normal_(self.C.data, gain=1.414)
        self.W = nn.Parameter(torch.zeros(size=(2 * hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.V = nn.Parameter(torch.zeros(size=(hidden_dim, num_class)))
        nn.init.xavier_normal_(self.V.data, gain=1.414)
        self.num_class = num_class
        self.hidden_dim = hidden_dim
    def forward(self,features):
        global neighbor_list, temp_out, x_1, x_2
        h = torch.mm(features, self.C)
        N = features.shape[0]
        outs = torch.zeros(size=(N, self.hidden_dim))
        for idx, neighbor_dict in self.weight_dict.items():
            if len(neighbor_dict) == 1:
                neighbor_list = list(neighbor_dict.keys())
                neighbor_list.append(neighbor_list[0])
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = h[node1_idx, :].reshape(1, -1)
                x_2 = h[node2_idx, :].reshape(1, -1)
                x = torch.cat([x_1, x_2], dim=1)
                temp_out = torch.mm(x, self.W)
            elif len(neighbor_dict) == 2:
                neighbor_list = list(neighbor_dict.keys())
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = h[node1_idx, :].reshape(1, -1)
                x_2 = h[node2_idx, :].reshape(1, -1)
                x = torch.cat([x_1, x_2], dim=1)
                temp_out = torch.mm(x, self.W)
            else:
                count = 10000
                temp_dict = {}
                neighbor_list = sortNodes(neighbor_dict)
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = h[node1_idx, :].reshape(1, -1)
                x_2 = h[node2_idx, :].reshape(1, -1)
                x = torch.cat([x_1,x_2],dim=1)
                temp_out = torch.mm(x,self.W)
                temp_dict[count] = temp_out
                temp_neighbor_dict = {key:weight for key,weight in neighbor_dict.items() if key != node1_idx and key != node2_idx}
                value = neighbor_dict[node1_idx] + neighbor_dict[node2_idx]
                temp_neighbor_dict[count] = value
                count += 1
                while len(temp_neighbor_dict) < 2:
                    temp_neighbor_list = sortNodes(temp_neighbor_dict)
                    node1_idx = temp_neighbor_list[0]
                    node2_idx = temp_neighbor_list[1]
                    if node1_idx >= 10000:
                        x_1 = temp_dict[node1_idx].reshape(1, -1)
                    else:
                        x_1 = h[node1_idx, :].reshape(1, -1)
                    if node2_idx >= 10000:
                        x_2 = temp_dict[node2_idx].reshape(1, -1)
                    else:
                        x_2 = h[node2_idx, :].reshape(1, -1)
                    x = torch.cat([x_1,x_2],dim=1)
                    temp_out = torch.mm(x,self.W)
                    temp_dict[count] = temp_out
                    value = temp_neighbor_dict[node1_idx] + temp_neighbor_dict[node2_idx]
                    temp_neighbor_dict = {key: weight for key, weight in temp_neighbor_dict.items() if
                                          key != node2_idx and key != node1_idx}
                    temp_neighbor_dict[count] = value
                    count += 1
                temp_out = temp_dict[count-1]
            outs[idx] = temp_out

        results = self.leakyrule(torch.mm(outs, self.V))

        return F.log_softmax(results, dim=1)


