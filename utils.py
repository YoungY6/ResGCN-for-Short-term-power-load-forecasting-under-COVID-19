import numpy as np
def getbackdata(inlist, st, dataout=24):
    reallist = list(inlist[:dataout])
    realistcount = [1] * dataout
    datalong = int(len(inlist) / dataout)
    for i in range(1, datalong):
        temlist = list(inlist)[i * dataout:(i + 1) * dataout]
        for j in range(st):
            reallist.append(temlist[-(st - j)])
            realistcount.append(1)
        for j in range(dataout - st):
            reallist[i * st + j] = reallist[i * st + j] + temlist[j]
            realistcount[i * st + j] = realistcount[i * st + j] + 1
    reallistout = []
    for r, c in zip(reallist, realistcount):
        tem = r / c
        reallistout.append(tem)
    return reallistout
def RMSE(data1, data2):
    data1, data2 = np.array(data1), np.array(data2)
    subdata = np.power(data1 - data2, 2)
    return np.sqrt(np.sum(subdata) / len(subdata - 1))
def MAPE(data1, data2):
    data1, data2 = np.array(data1), np.array(data2)
    data1 = np.squeeze(data1)
    data2 = np.squeeze(data2)
    MAPE_mid = abs((data1 - data2) / data1)
    MAPE = sum(MAPE_mid) / len(MAPE_mid)
    return MAPE
