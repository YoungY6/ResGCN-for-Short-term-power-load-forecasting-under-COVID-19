from easydict import EasyDict as edict
import json
import os
config = edict()

# Pretreatment and analysis
config.Pretreatment = edict()
config.Pretreatment.data_ori='data_ori'
config.Pretreatment.data_analysis='data_analysis'
if not os.path.exists(config.Pretreatment.data_analysis):
    os.mkdir(config.Pretreatment.data_analysis)

config.Pretreatment.market='ercot'
config.Pretreatment.city='houston'

config.Pretreatment.data_start_date1='2020-01-23'
config.Pretreatment.data_start_date2='2019-01-01'
config.Pretreatment.data_end_date1='2020-11-22'
config.Pretreatment.data_end_date2='2019-12-31'

config.Pretreatment.cityelement_covid_happened=['load','weather','covid','patterns','social_distancing']
config.Pretreatment.cityelement_no_covid=['load','weather']
# train model
config.Train = edict()
config.Train.model_dataset='model_dataset'
if not os.path.exists(config.Train.model_dataset):
    os.mkdir(config.Train.model_dataset)
config.Train.modeldir='model'
if not os.path.exists(config.Train.modeldir):
    os.mkdir(config.Train.modeldir)
config.Train.inputnum=48
config.Train.outputnum=24
config.Train.huastep=24
config.Train.traindata_rate=0.8
config.Train.validdata_rate=0.1
config.Train.testdata_rate=0.1

config.Train.batchsize=118
config.Train.epoch=1200

config.Train.chose_factor=['load','new_confirm','Grocery_Pharmacy','completely_home_device_count_percentage','weather','preload']
config.Train.pic='pic'
if not os.path.exists(config.Train.pic):
    os.mkdir(config.Train.pic)