# ResGCN-for-Short-term-power-load-forecasting-under-COVID-19

# Directory introduction
——**"data_ori"** is the datasource from https://github.com/tamu-engineering-research/COVID-EMDA  

——**"data_analysis"** is the outcome of extract_data from "data_ori", you can run extract_data.py to make this data  

——**"model"** is saving the ResGCN model, there are two models and you can run the 'testmodel.py' to get the test results,already two models that perform well

——**'model_dataset'** is the files for training the model.

——**'pic'** is the results of running 'testmodel.py',you can clearly see the results of the model.

# Quick start
1. ##### run the extract_data.py.  ——you can extract the original data to 'data_analysis'
2. ##### run the mkdataset.py.  ——you can make the dataset for training and testing ,you can see the dataset in 'model_dataset'
3. ##### run the trainmodel.py.  ——you can trarin the ResGCN model, and you can see the new model in 'model'
4. ##### run the testmodel.py.  ——you can see the results of ResGCN, at the same time some data charts will be displayed, you can see there in the 'pic'




