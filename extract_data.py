import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calendar
import datetime
import random
from config import config
random.seed(0)


# This function is very important to extract data
def getdatalist(df,colitem=''):
    if colitem=='':
        if 'kind' in df.columns:
            del df['kind']
        outlist = []
        for i in range(len(df)):
            for h in range(24):
                outlist.append(df.iloc[i, int(1 + h)])
        datamax=max(outlist)
        datamin=min(outlist)
        # outlist_nor = (outlist - datamin) / (datamax - datamin)
        outlist_nor = (outlist - datamin) / (datamax - datamin)
    else:
        outlist = []
        for i in range(len(df)):
            for h in range(24):
                outlist.append(df[colitem].iloc[i])
        datamax=max(outlist)
        datamin=min(outlist)
        outlist_nor = (outlist - datamin) / (datamax - datamin)
    # print(len(outlist))
    return {'data':outlist_nor,
            'datamax':datamax,'datamin':datamin}
def extract_data(parameter):
    # read the parameter
    startdate=pd.to_datetime(parameter['startdate'])
    enddate=pd.to_datetime(parameter['enddate'])
    data_ori=parameter['data_ori']
    cityelement=parameter['cityelement']
    data_analysis=parameter['data_analysis']
    market=parameter['market']
    city=parameter['city']

    # mk complete datelist, because some dates are missing
    datelist=[]
    num=0
    while startdate+datetime.timedelta(days=num)<=enddate:
        datelist.append(startdate+datetime.timedelta(days=num))
        num=num+1
    df_dict = {}
    for choseelement in cityelement: # read all csv ，and get a data json
        dftem=pd.read_csv(os.path.join(data_ori,market+'_'+city+'_'+choseelement+'.csv'))
        dftem['date']=pd.to_datetime(dftem['date'])
        dftem=dftem[(dftem['date']>=startdate)&
                    (dftem['date']<=enddate)]
        dftem_out=pd.DataFrame()
        for day in range(len(datelist)):
            if len(dftem[dftem['date']==datelist[day]])==1:
                dftem_out=dftem_out.append(dftem[dftem['date']==datelist[day]])
            else:
                dataitem={
                    'date':datelist[day]
                }
                col= dftem.columns
                if 'date' in col:
                    col=col.drop('date')
                for c in col:
                    try:
                        dataitem[c]=(dftem[dftem['date']==datelist[day-1]][c].iloc[0]+
                                 dftem[dftem['date']==datelist[day+1]][c].iloc[0])/2
                    except:
                        dataitem[c] = dftem[dftem['date'] == datelist[day - 1]][c].iloc[0]
                dataitem=pd.DataFrame([dataitem])
                dftem_out = dftem_out.append(dataitem)
        df_dict[choseelement]=dftem_out


    dfmap = pd.DataFrame() # this is output dataframe
    # dfmap['date']=datelist
    """extract data"""

    # load
    load=getdatalist(df_dict['load'])
    dfmap['load'] = load['data']


    """covid"""
    try:
        # covid
        covid=getdatalist(df_dict['covid'],'accum_confirm')
        dfmap['covid'] = covid['data']
    except:
        pass
    try:
        # new_confirm
        new_confirm=getdatalist(df_dict['covid'],'new_confirm')
        dfmap['new_confirm'] = new_confirm['data']
    except:
        pass
    try:
        # new_confirm
        infect_rate=getdatalist(df_dict['covid'],'infect_rate')
        dfmap['infect_rate'] = infect_rate['data']
    except:
        pass
    try:
        # accum_death
        accum_death=getdatalist(df_dict['covid'],'accum_death')
        dfmap['accum_death'] = accum_death['data']
    except:
        pass
    try:
        # new_death
        new_death=getdatalist(df_dict['covid'],'new_death')
        dfmap['new_death'] = new_death['data']
    except:
        pass
    try:
        # fatal_rate
        fatal_rate=getdatalist(df_dict['covid'],'fatal_rate')
        dfmap['fatal_rate'] = fatal_rate['data']


    except:
        pass
    """pattern"""
    try:
    # Restaurant_Recreaction
        Restaurant_Recreaction=getdatalist(df_dict['patterns'],'Restaurant_Recreaction')
        dfmap['Restaurant_Recreaction'] = Restaurant_Recreaction['data']
    except:
        pass
    try:
    # Grocery_Pharmacy
        Grocery_Pharmacy=getdatalist(df_dict['patterns'],'Grocery_Pharmacy')
        dfmap['Grocery_Pharmacy'] = Grocery_Pharmacy['data']
    except:
        pass
    try:
    # Retail
        Retail=getdatalist(df_dict['patterns'],'Retail')
        dfmap['Retail'] = Retail['data']
    except:
        pass

    """social_distancing"""
    try:
    # median_home_dwell_time_percentage
        median_home_dwell_time_percentage=getdatalist(df_dict['social_distancing'],'median_home_dwell_time_percentage')
        dfmap['median_home_dwell_time_percentage'] = median_home_dwell_time_percentage['data']
    except:
        pass
    try:
    # part_time_work_behavior_devices_percentage
        part_time_work_behavior_devices_percentage=getdatalist(df_dict['social_distancing'],'part_time_work_behavior_devices_percentage')
        dfmap['part_time_work_behavior_devices_percentage'] = part_time_work_behavior_devices_percentage['data']
    except:
        pass

    try:
    # full_time_work_behavior_devices_percentage
        full_time_work_behavior_devices_percentage=getdatalist(df_dict['social_distancing'],'full_time_work_behavior_devices_percentage')
        dfmap['full_time_work_behavior_devices_percentage'] = full_time_work_behavior_devices_percentage['data']
    except:
        pass

    try:
    # completely_home_device_count
        completely_home_device_count=getdatalist(df_dict['social_distancing'],'completely_home_device_count')
        dfmap['completely_home_device_count'] = completely_home_device_count['data']
    except:
        pass

    try:
    # device_count
        device_count=getdatalist(df_dict['social_distancing'],'device_count')
        dfmap['device_count'] = device_count['data']
    except:
        pass

    try:
        # completely_home_device_count_percentage
        completely_home_device_count_percentage=getdatalist(df_dict['social_distancing'],'completely_home_device_count_percentage')
        dfmap['completely_home_device_count_percentage'] = completely_home_device_count_percentage['data']
    except:
        pass

    try:
        # completely_home_device_count_percentage
        part_time_work_behavior_devices=getdatalist(df_dict['social_distancing'],'part_time_work_behavior_devices')
        dfmap['part_time_work_behavior_devices'] = part_time_work_behavior_devices['data']
    except:
        pass
    try:
        # completely_home_device_count_percentage
        full_time_work_behavior_devices=getdatalist(df_dict['social_distancing'],'full_time_work_behavior_devices')
        dfmap['full_time_work_behavior_devices'] = full_time_work_behavior_devices['data']
    except:
        pass
    """weather"""
    try:
        # weather
        weather=getdatalist(df_dict['weather'])
        dfmap['weather']=weather['data']
    except:
        pass
    dfmap.to_excel(os.path.join(data_analysis,
                                market+'_'+city+parameter['startdate']+'_'+'to'+'_'+parameter['enddate']+'_extractdata.xlsx'),
                   index=False)
parameter1={
    'data_ori': config.Pretreatment.data_ori,
    'startdate':config.Pretreatment.data_start_date1,
    'enddate':config.Pretreatment.data_end_date1,
    'cityelement':config.Pretreatment.cityelement_covid_happened,
    'data_analysis':config.Pretreatment.data_analysis,
    'market':config.Pretreatment.market,
    'city':config.Pretreatment.city,
}
extract_data(parameter1)
parameter2={
    'data_ori': config.Pretreatment.data_ori,
    'startdate':config.Pretreatment.data_start_date2,
    'enddate':config.Pretreatment.data_end_date2,
    'cityelement':config.Pretreatment.cityelement_no_covid,
    'data_analysis':config.Pretreatment.data_analysis,
    'market':config.Pretreatment.market,
    'city':config.Pretreatment.city,
}
extract_data(parameter2)
