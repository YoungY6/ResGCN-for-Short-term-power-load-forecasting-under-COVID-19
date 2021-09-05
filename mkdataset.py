import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calendar
import datetime
import random
from config import config
random.seed(0)


def mkdataset(parameter):
    model_dataset = parameter['model_dataset']
    cityelement = parameter['cityelement']
    data_analysis = parameter['data_analysis']
    market = parameter['market']
    city = parameter['city']
    inputnum = parameter['inputnum']
    outputnum = parameter['outputnum']
    huastep = parameter['huastep']
    traindata_rate = parameter['traindata_rate']
    validdata_rate = parameter['validdata_rate']
    testdata_rate = parameter['testdata_rate']


    dfmap=pd.read_excel(os.path.join(data_analysis,
                                     market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'_extractdata.xlsx'))

    dfmapout=pd.DataFrame()
    mapcol=dfmap.columns
    hua = 0
    for i in range(int((len(dfmap)-inputnum-outputnum)/huastep)):
        dfitemdict={}
        for c in mapcol:
            for j in range(inputnum):
                dfitemdict[c+str(j+1)]=dfmap[c].iloc[j+hua]
        for j in range(outputnum):
            dfitemdict['preload' + str(j + 1)] = dfmap['load'].iloc[j + inputnum + hua]
        dfitem_df=pd.DataFrame([dfitemdict])
        dfmapout=dfmapout.append(dfitem_df)
        hua=hua+huastep

    datalen=len(dfmapout)
    dfmaptrain=dfmapout.iloc[:int(datalen*traindata_rate)]
    dfmapvalid=dfmapout.iloc[int(datalen*traindata_rate):int(datalen*(traindata_rate+validdata_rate))]
    dfmaptest=dfmapout.iloc[int(datalen*(traindata_rate+validdata_rate)):]

    dfmapout.to_csv(os.path.join(model_dataset,str(huastep) + market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+ 'allmapdata.csv'),
                      index=False, encoding='utf-8')
    dfmaptrain.to_csv(os.path.join(model_dataset,str(huastep)+market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'trainmapdata.csv'),
                      index=False,encoding='utf-8')
    dfmapvalid.to_csv(os.path.join(model_dataset,str(huastep)+market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'validmapdata.csv'),
                      index=False,encoding='utf-8')
    dfmaptest.to_csv(os.path.join(model_dataset,str(huastep)+market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'testmapdata.csv'),
                     index=False,encoding='utf-8')
parameter1={
    'startdate':config.Pretreatment.data_start_date1,
    'enddate':config.Pretreatment.data_end_date1,
    'cityelement':config.Pretreatment.cityelement_covid_happened,
    'data_analysis':config.Pretreatment.data_analysis,
    'market':config.Pretreatment.market,
    'city':config.Pretreatment.city,
    'model_dataset':config.Train.model_dataset,
    'inputnum':config.Train.inputnum,
    'outputnum':config.Train.outputnum,
    'huastep':config.Train.huastep,
    'traindata_rate':config.Train.traindata_rate,
    'validdata_rate':config.Train.validdata_rate,
    'testdata_rate':config.Train.testdata_rate,
}
mkdataset(parameter1)

parameter2={
    'startdate':config.Pretreatment.data_start_date2,
    'enddate':config.Pretreatment.data_end_date2,
    'cityelement':config.Pretreatment.cityelement_no_covid,
    'data_analysis':config.Pretreatment.data_analysis,
    'market':config.Pretreatment.market,
    'city':config.Pretreatment.city,
    'model_dataset':config.Train.model_dataset,
    'inputnum':config.Train.inputnum,
    'outputnum':config.Train.outputnum,
    'huastep':config.Train.huastep,
    'traindata_rate':config.Train.traindata_rate,
    'validdata_rate':config.Train.validdata_rate,
    'testdata_rate':config.Train.testdata_rate,
}
mkdataset(parameter2)