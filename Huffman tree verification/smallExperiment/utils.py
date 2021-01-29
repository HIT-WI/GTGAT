#encoding=utf-8


import numpy as np
import scipy.sparse as sp
import torch
import random
import pickle

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum,-0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get,labels)),dtype=np.int32)
    return labels_onehot

def load_ori_data(filepath,dataset):
    print("Loding {} ori dataset... ".format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(filepath,dataset),dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:,-1])

    idx = np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(filepath,dataset),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    return adj,features,labels
def getweightDicts(filepath,idx_dict):
    adj_dicts = {}
    with open(filepath,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        cnt = adj_dicts.get(line[0],-1)
        if cnt == -1:
            adj_dicts[line[0]] = [line[1]]
        else:
            if line[1] not in adj_dicts[line[0]]:
                adj_dicts[line[0]].append(line[1])
        cnt = adj_dicts.get(line[1],-1)
        if cnt == -1:
            adj_dicts[line[1]] = [line[0]]
        else:
            if line[0] not in adj_dicts[line[1]]:
                adj_dicts[line[1]].append(line[0])
    weight_dict = {}
    for key,value in adj_dicts.items():
        for item in value:
            cnt = weight_dict.get(key,-1)
            if cnt == -1:
                weight_dict[key] = {}
            weight_dict[key][item] = 1.0
    for key,value in weight_dict.items():
        for item in value:
            for key1,value1 in weight_dict.items():
                if key in value1 and item in value1 and key != key1 and item != key1:
                    weight_dict[key][item] += 1.0
                    weight_dict[item][key] += 1.0
        weight_dict[key][key] = 100.0
    final_dict = {}
    for key,value in weight_dict.items():
        final_dict[idx_dict[key]] = {}
        for item,num in value.items():
            final_dict[idx_dict[key]][idx_dict[item]] = num
    return final_dict

def getEntitiesID(filepath):
    nums = []
    with open(filepath,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            nums.append(line[0])
    idx_dict = {}
    for idx,id in enumerate(nums):
        idx_dict[id] = idx
    return idx_dict

def getAdjDict(filepath,idx_dict):
    adj_dicts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        cnt = adj_dicts.get(line[0], -1)
        if cnt == -1:
            adj_dicts[line[0]] = [line[1]]
        else:
            if line[1] not in adj_dicts[line[0]]:
                adj_dicts[line[0]].append(line[1])
        cnt = adj_dicts.get(line[1], -1)
        if cnt == -1:
            adj_dicts[line[1]] = [line[0]]
        else:
            if line[0] not in adj_dicts[line[1]]:
                adj_dicts[line[1]].append(line[0])
    adj_dict = {}
    for key,value in adj_dicts.items():
        adj_dict[idx_dict[key]] = []
        for item in value :
            adj_dict[idx_dict[key]].append(idx_dict[item])
    return adj_dict




def sortNodes(weight_dict):
    weight_dict = sorted(weight_dict.items(),key=lambda x:x[1],reverse=False)
    node_list = [item[0] for item in weight_dict]
    return node_list


def makedata():
    with open("./data/chose_fea.pkl","rb") as f:
        chose_features = pickle.load(f)
    with open("./data/chose_label.pkl","rb") as f:
        chose_labels = pickle.load(f)
    with open("./data/chose_weight_dict.pkl","rb") as f:
        chose_weight_dict = pickle.load(f)
    idx_train = range(60)
    idx_val = range(100, 200)
    idx_test = range(100, 250)
    return chose_features, chose_labels, chose_weight_dict, idx_train, idx_val, idx_test

def accuracy(output,labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)











if __name__ == "__main__":
    adj,features,labels = load_ori_data('data/cora/', 'cora')
    idx_dict = getEntitiesID("data/cora/cora.content")
    adj_dict = getAdjDict("data/cora/cora.cites",idx_dict)
    weight_dict = getweightDicts("./data/cora/cora.cites",idx_dict)
    chose_features,chose_labels,chose_weight_dict = choseDataset(features, labels,adj_dict,weight_dict)
    print(chose_weight_dict)