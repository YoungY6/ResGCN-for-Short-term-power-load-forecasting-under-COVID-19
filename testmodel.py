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
import matplotlib.pyplot as plt
import matplotlib
import os
from config import config
from utils import *
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
picpath = config.Train.pic
modeldir = config.Train.modeldir
parameter={
    'startdate':config.Pretreatment.data_start_date1,
    'enddate':config.Pretreatment.data_end_date1,
}
#%%
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
model.load_state_dict(torch.load(os.path.join(modeldir,str(huastep)+market+city+'model_notadd.pth'),map_location=device))  # 加载参数
testpath1 = os.path.join(model_dataset,str(huastep)+market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'testmapdata.csv')
colstart=[]
dftem=pd.read_csv(testpath1)
collist=dftem.columns
collist=list(collist)
cc_s=0
while cc_s<len(collist):
    if collist[cc_s][:-1] in chose_factor:
        colstart.append(cc_s)
        cc_s=cc_s+48
    else:
        cc_s=cc_s+1
class Dataprovider_t1(object):
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
        # row = list(range(len(data)))
        # random.shuffle(row)
        process_data = data[:, self.column]
        # process_data = process_data[row, :]
        x = process_data[:, list(range(48))]
        x = x.reshape(len(data),-1,1)
        y = process_data[:, list(range(48, 72))]
        for i in range(0, len(process_data), batchsize):
            input = x[i:i + batchsize, :]
            output = y[i:i + batchsize, :]
            yield input, output
dataprovider_test1_1 = Dataprovider_t1(testpath1, batchsize)
losses = []
best_valid = np.inf
print('start testing not add !')
model.eval()
output_container1 = []
truth_container1 = []
with torch.no_grad():
    for x, y in dataprovider_test1_1.feed_chunk():
        x = x.reshape(-1, 1)
        # print(np.shape(x))
        x = torch.FloatTensor(x).to(device)
        output = model(x)
        output_array = np.array(output.cpu()).reshape(-1, 1)
        output_container1.append(output_array)
        truth_container1.append(y.reshape(-1, 1))
    pre = np.vstack(output_container1)
    gru = np.vstack(truth_container1)
    a = 1
for i in range(10):
    plt.plot(pre[i * 24:(i + 1) * 24], color='b', label='pre')
    plt.plot(gru[i * 24:(i + 1) * 24], color='r', label='real')
    plt.legend()
    plt.savefig(os.path.join(picpath,'notadd' + str(i) + '.png'))
    plt.clf()
reallistback = getbackdata(gru, huastep)
prelistback = getbackdata(pre, huastep)
plt.clf()
plt.plot(prelistback, color='b', label='pre')
plt.plot(reallistback, color='r', label='real')
plt.legend()
plt.savefig(os.path.join(picpath, 'notadd_alldatashow' + '.png'))
plt.clf()

#%%
numaffect = 5
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
        out = self.channel_attention(x) * x  # (1,16,1,1)*x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(1, 32)
        self.conv1 = GCN2Conv(32, 0.1, 0.5, 1)
        self.conv2 = GCN2Conv(32, 0.1, 0.5, 2)
        self.conv3 = GCN2Conv(32, 0.1, 0.5, 4)
        self.conv4 = GCN2Conv(32, 0.1, 0.5, 8)
        self.lin0 = Linear(numaffect, 32)
        self.lstm = nn.LSTM(32, 118, 2, batch_first=True, bidirectional=True)
        self.cbam = CBAM(channel=118 * 2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.lin1 = Linear(118 * 2 * int(winlen / 2), 24)
        self.lin2 = Linear(24, 24)
        # self.lin3 = Linear(64, 24)

    def forward(self, data):
        x0 = self.lin0(data)
        x = F.relu(x0)
        x = self.conv1(x, x0, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, x0, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, x0, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, x0, edge_index, edge_attr)
        x = F.relu(x)
        x = x.reshape(int(x.size(0) / winlen), winlen, 32)
        out, (h_n, c_n) = self.lstm(x)  # out是
        # x = h_n[-1, :, :]
        x = out
        x = x.reshape(x.size(0), 118 * 2, winlen, 1)
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
modelpath='model'+'both118_4c_2l_allzb'
model.load_state_dict(torch.load(
    # os.path.join('model','houston'+str(24)+'model_add.pth'),
    os.path.join(modeldir,str(huastep)+market+city+'model_add.pth'),
    # '/home/jiajun/codework/map/'+modelpath+'/'+'houston'+str(24)+'model_add.pth',
    map_location=device))  # 加载参数
class Dataprovider_t2(object):
    def __init__(self, trainpath, batchsize):
        column = []
        for i in range(48):
            for j in range(numaffect):
                column.append(i + colstart[j])
        for i in range(24):
            column.append(colstart[-1] + i)
        self.batchsize = batchsize
        self.trainpath = trainpath
        self.column = column

    def feed_chunk(self):
        tem = pd.read_csv(self.trainpath)
        data = np.array(tem)
        # row = list(range(len(data)))
        # random.shuffle(row)
        process_data = data[:, self.column]
        # process_data = process_data[row, :]
        x = process_data[:, list(range(48 * numaffect))]
        x = x.reshape(len(data), -1, numaffect)
        y = process_data[:, list(range(48 * numaffect, 48 * numaffect + 24))]
        for i in range(0, len(process_data), batchsize):
            input = x[i:i + batchsize, :]
            output = y[i:i + batchsize, :]
            yield input, output
dataprovider_test1_2 = Dataprovider_t2(testpath1, batchsize)
losses = []
best_valid = np.inf
print('start testing add factor!')
model.eval()
output_container2 = []
truth_container2 = []
with torch.no_grad():
    for x, y in dataprovider_test1_2.feed_chunk():
        x = x.reshape(-1, numaffect)
        # print(np.shape(x))
        x = torch.FloatTensor(x).to(device)
        output = model(x)
        output_array = np.array(output.cpu()).reshape(-1, 1)
        output_container2.append(output_array)
        truth_container2.append(y.reshape(-1, 1))
    pre2 = np.vstack(output_container2)
    gru2 = np.vstack(truth_container2)
    a = 1
for i in range(10):
    plt.plot(pre2[i * 24:(i + 1) * 24], color='b', label='pre')
    plt.plot(gru2[i * 24:(i + 1) * 24], color='r', label='real')
    plt.legend()
    plt.savefig(os.path.join(picpath,'add' + str(i) + '.png'))
    plt.clf()
reallistback2 = getbackdata(gru2, huastep)
prelistback2 = getbackdata(pre2, huastep)
# %%
plt.clf()
plt.plot(prelistback2, color='b', label='pre')
plt.plot(reallistback2, color='r', label='real')
plt.legend()
plt.savefig(os.path.join(picpath,'add_alldatashow' + '.png'))
# %% both line
plt.clf()
plt.plot(prelistback2, color='b', label='pre2_add')
plt.plot(prelistback, color='y', label='pre1_notadd')
plt.plot(reallistback2, color='r', label='real')
plt.legend()
plt.savefig(os.path.join(picpath,'both_alldatashow' + '.png'))
# %%
plt.clf()
plt.plot(prelistback2[0:72], color='b', label='pre2_add')
plt.plot(prelistback[0:72], color='y', label='pre1_notadd')
plt.plot(reallistback2[0:72], color='r', label='real')
plt.legend()
plt.savefig(os.path.join(picpath, 'both_threedays_datashow' + '.png'))
# %%
plt.clf()
for i in range(10):
    plt.plot(pre2[i * 24:(i + 1) * 24], color='b', label='pre_add')
    plt.plot(pre[i * 24:(i + 1) * 24], color='y', label='pre_noadd')
    plt.plot(gru2[i * 24:(i + 1) * 24], color='r', label='real')
    plt.legend()
    plt.savefig(os.path.join(picpath, 'compare' + str(i) + '.png'))
    plt.clf()
RMSE_add = RMSE(reallistback2, prelistback2)

RMSE_notadd = RMSE(reallistback2, prelistback)

MAPE_add = MAPE(reallistback2, prelistback2)

MAPE_notadd = MAPE(reallistback2, prelistback)

print('----------------------------')
print('RMSE_add:', RMSE_add)
print('RMSE_notadd:', RMSE_notadd)
print('MAPE_add:', MAPE_add)
print('MAPE_notadd:', MAPE_notadd)



