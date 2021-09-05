import pandas as pd
import random
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, APPNP, SAGEConv, GCN2Conv,GATConv
from torch.nn import Linear
import os
from config import config
random.seed(0)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batchsize =config.Train.batchsize
winlen=config.Train.inputnum
offset = int(0.5*(winlen))
def timeweight(x1,x2):
    return np.exp(-abs(x1-x2)/30)
soucenode,endnode,edgefeature= [],[],[]
for i in range(winlen):
    for j in range(winlen-1):
        soucenode.append(i)
for i in range(winlen):
    s = list(range(winlen))
    s.remove(i)
    endnode = endnode + s
endnode = list(endnode)
soucenode = soucenode * batchsize
endnode = endnode * batchsize
endnode = np.array(endnode)
soucenode = np.array(soucenode)
edge_index = torch.tensor([soucenode, endnode], dtype=torch.long).to(device)
for row in range(winlen ):
    for col in range(winlen):
        if row != col:
            edgefeature.append(timeweight(row, col))
edgefeature = np.array(edgefeature)
edgefeature = edgefeature / np.max(edgefeature)
aa = list(edgefeature)
edgefeature = aa * batchsize
edgefeature = torch.FloatTensor(edgefeature)
edge_attr = edgefeature.to(device)
#%%
model_dataset = config.Train.model_dataset
city = config.Pretreatment.city
market = config.Pretreatment.market
chose_factor = config.Train.chose_factor
huastep = config.Train.huastep
ep = config.Train.epoch
modeldir = config.Train.modeldir
parameter={
    'startdate':config.Pretreatment.data_start_date1,
    'enddate':config.Pretreatment.data_end_date1,
}
trainpath = os.path.join(model_dataset,str(huastep)+market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'trainmapdata.csv')
colstart=[]
dftem=pd.read_csv(trainpath)
collist=dftem.columns
collist=list(collist)
cc_s=0
while cc_s<len(collist):
    if collist[cc_s][:-1] in chose_factor:
        colstart.append(cc_s)
        cc_s=cc_s+48
    else:
        cc_s=cc_s+1
"""not add"""
class Dataprovider(object):
    def __init__(self, trainpath, batchsize):
        column = []
        for i in range(48):
            column.append(i)
        for i in range(24):
            column.append(colstart[-1] + i)
        self.batchsize = batchsize
        self.trainpath = trainpath
        self.column = column

    def feed_chunk(self):
        tem=pd.read_csv(self.trainpath)
        data = np.array(tem)
        row = list(range(len(data)))
        random.shuffle(row)
        process_data = data[:, self.column]
        process_data = process_data[row, :]
        x = process_data[:, list(range(48))]
        x = x.reshape(len(data),-1,1)
        y = process_data[:, list(range(48, 72))]
        for i in range(0, len(process_data), batchsize):
            input = x[i:i + batchsize, :]
            output = y[i:i + batchsize, :]
            yield input, output
dataprovider = Dataprovider(trainpath,batchsize)
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x   #(1,16,1,1)*x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(1, 32)
        self.conv1 = GCN2Conv(32,0.1,0.5,1)
        self.conv2 = GCN2Conv(32,0.1,0.5,2)
        self.conv3 = GCN2Conv(32,0.1,0.5,2)
        self.conv4 = GCN2Conv(32,0.1,0.5,2)
        self.lin0 =Linear(1,32)
        self.lstm = nn.LSTM(32, 118, 2, batch_first=True,bidirectional=True)
        self.cbam = CBAM(channel=118*2)
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.lin1 = Linear(118*2*int(winlen/2),24)
        self.lin2 = Linear(24, 24)
        # self.lin3 = Linear(64, 24)


    def forward(self, data):

        x0 = self.lin0(data)
        x = F.relu(x0)
        x = self.conv1(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = x.reshape(int(x.size(0)/winlen),winlen,32)
        out, (h_n, c_n) = self.lstm(x)  #out是
        # x = h_n[-1, :, :]
        x = out
        x = x.reshape(x.size(0), 118*2,winlen, 1)
        x = self.cbam(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        # x = self.lin3(x)
        # x = F.relu(x)
        return x
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.1)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, torch_geometric.nn.GCNConv):
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.weight, 0.5)
        torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
model = Net().to(device)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
Loss_F = torch.nn.MSELoss()
losses = []
best_valid = np.inf
print('start training model which not add factor!')
for epoch in range(ep):
    model.train()
    trainlosstotal = 0
    for step, (x, y) in enumerate(dataprovider.feed_chunk()):
        # y = y.reshape(x.shape[0])
        y = torch.FloatTensor(y).to(device)
        data = x.reshape(-1,1)
        data = torch.FloatTensor(data)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(np.shape(output))
        # print(np.shape(y))
        loss = Loss_F(output, y)
        loss.backward()
        optimizer.step()
        trainlosstotal = trainlosstotal + loss.item()
    print('epoch: {}, trainLoss: {:.4f}'.format(epoch, trainlosstotal /  step))
    losses.append(trainlosstotal /  step)
torch.save(model.state_dict(), os.path.join(modeldir,str(huastep)+market+city+'model_notadd.pth'))  # save model
"""add factor"""
numaffect=5
class Dataprovider(object):
    def __init__(self, trainpath, batchsize):
        column = []
        for i in range(48):
            for j in range(numaffect):
                column.append(i+colstart[j])
        for i in range(24):
            column.append(colstart[-1] + i)
        self.batchsize = batchsize
        self.trainpath = trainpath
        self.column = column

    def feed_chunk(self):
        tem=pd.read_csv(self.trainpath)
        data = np.array(tem)
        row = list(range(len(data)))
        random.shuffle(row)
        process_data = data[:, self.column]
        process_data = process_data[row, :]
        x = process_data[:, list(range(48*numaffect))]
        x = x.reshape(len(data),-1,numaffect)
        y = process_data[:, list(range(48*numaffect, 48*numaffect+24))]
        for i in range(0, len(process_data), batchsize):
            input = x[i:i + batchsize, :]
            output = y[i:i + batchsize, :]
            yield input, output
dataprovider = Dataprovider(trainpath,batchsize)
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x   #(1,16,1,1)*x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(1, 32)
        self.conv1 = GCN2Conv(32,0.1,0.5,1)
        self.conv2 = GCN2Conv(32,0.1,0.5,2)
        self.conv3 = GCN2Conv(32,0.1,0.5,4)
        self.conv4 = GCN2Conv(32,0.1,0.5,8)
        self.lin0 =Linear(numaffect,32)
        self.lstm = nn.LSTM(32, 118, 2, batch_first=True,bidirectional=True)
        self.cbam = CBAM(channel=118*2)
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.lin1 = Linear(118*2*int(winlen/2),24)
        self.lin2 = Linear(24, 24)
        # self.lin3 = Linear(64, 24)


    def forward(self, data):

        x0 = self.lin0(data)
        x = F.relu(x0)
        x = self.conv1(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, x0,edge_index, edge_attr)
        x = F.relu(x)
        x = x.reshape(int(x.size(0)/winlen),winlen,32)
        out, (h_n, c_n) = self.lstm(x)  #out是
        # x = h_n[-1, :, :]
        x = out
        x = x.reshape(x.size(0), 118*2,winlen, 1)
        x = self.cbam(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        # x = self.lin3(x)
        # x = F.relu(x)
        return x
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.1)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, torch_geometric.nn.GCNConv):
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.weight, 0.5)
        torch.nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
model = Net().to(device)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
Loss_F = torch.nn.MSELoss()
losses = []
best_valid = np.inf
print('start training model which add factor!')
for epoch in range(ep):
    model.train()
    trainlosstotal = 0
    for step, (x, y) in enumerate(dataprovider.feed_chunk()):
        y = torch.FloatTensor(y).to(device)
        data = x.reshape(-1,numaffect)
        data = torch.FloatTensor(data)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Loss_F(output, y)
        loss.backward()
        optimizer.step()
        trainlosstotal = trainlosstotal + loss.item()
    print('epoch: {}, trainLoss: {:.4f}'.format(epoch, trainlosstotal /  step))
    losses.append(trainlosstotal /  step)
torch.save(model.state_dict(), os.path.join(modeldir,str(huastep)+market+city+'model_add.pth'))  # save model

