#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import json
from collections import Counter


def getdata():
    du_dict_self,du_dict_ci,du_dict_cora = [],[],[]
    with open("./data/adj_dict.json","r",encoding="utf-8") as f:
        adj_dict_o = json.load(f)
    with open("./data/adj_dict_ci.json","r",encoding="utf-8") as f:
        du_dict_ci = json.load(f)
    with open("./data/adj_dict_cora.json","r",encoding="utf-8") as f:
        du_dict_cora = json.load(f)
    for _,value in adj_dict_o.items():
        if len(value) == 0:
            du_dict_self.append(1)
        else:
            du_dict_self.append(len(value))
    dd = Counter(du_dict_self)
    du_dict_cora = Counter(du_dict_cora)
    du_dict_ci = Counter(du_dict_ci)
    x = list(dd.keys())
    y = list(dd.values())
    x_ci = list(du_dict_ci.keys())
    y_ci = list(du_dict_ci.values())
    x_co = list(du_dict_cora.keys())
    y_co = list(du_dict_cora.values())
    for i in range(len(x)):
        x[i] = math.log10(x[i])
        y[i] = math.log10(y[i])
    for i in range(len(x_ci)):
        x_ci[i] = math.log10(x_ci[i])
        y_ci[i] = math.log10(y_ci[i])
    for i in range(len(x_co)):
        x_co[i] = math.log10(x_co[i])
        y_co[i] = math.log10(y_co[i])

    x = torch.tensor(x).float().reshape(-1,1)
    y = torch.tensor(y).float().reshape(-1,1)
    x_ci = torch.tensor(x_ci).float().reshape(-1,1)
    x_co = torch.tensor(x_co).float().reshape(-1,1)
    y_ci = torch.tensor(y_ci).float().reshape(-1,1)
    y_co = torch.tensor(y_co).float().reshape(-1,1)
    return x,y,x_ci,y_ci,x_co,y_co

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(1,1)))
        nn.init.xavier_normal_(self.W.data,gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1,1)))
        nn.init.xavier_normal_(self.bias.data,gain=1.414)
    def forward(self,x):
      

        return torch.mm(x,self.W) + self.bias


loss_f = nn.MSELoss()
model = LinearRegression()
optimer = optim.SGD(model.parameters(),lr=0.01)
x,y,x_ci,y_ci,x_co,y_co = getdata()
print(x_co.shape,y_co.shape)

def train(x_value,y_value):

    best_loss = 1000
    outs = model(x_value)
    optimer.zero_grad()
    loss = loss_f(outs,y_value)
    loss.backward()
    optimer.step()
    count = 0
    while loss.data.item() < best_loss:
        outs = model(x_value)
        optimer.zero_grad()
        loss = loss_f(outs, y_value)
        loss.backward()
        optimer.step()
        count += 1
        if count >20000:
            break
        print("current loss is ", loss.data.item())
    print("the result W is ",model.W.data.item())
    return torch.mm(x_value,model.W),model.W



if __name__ == "__main__":
    outs = train(x_co,y_co)



