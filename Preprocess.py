import os
import warnings
from datetime import timedelta
import datetime
import numpy as np 
import pandas as pd

warnings.filterwarnings('ignore')

# input_dir='../input/aws-spot-price-15th-feb16th-april-2021/'
# US_Only=True

def pre_process(input_dir,US_Only=True,logging=False):
    if logging: print(str(datetime.datetime.now()).split('.')[0],'load data...')
    files=os.listdir(input_dir)
    if US_Only: 
        files=[x for x in files if 'us-' in x]

    df_us=pd.read_csv(input_dir+files[0])
    for file in files[1:]:
        df_us = pd.concat((df_us,pd.read_csv(input_dir+file)))

    df_us['Timestamp'] = pd.to_datetime(df_us['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f', utc=True)
    df_us = df_us.sort_values(by="Timestamp")

    # price add on
    if logging: print(str(datetime.datetime.now()).split('.')[0],'calculating price add on ...')
    data=df_us.groupby(['AvailabilityZone','InstanceType','ProductDescription'],as_index=False)['SpotPrice'].min()
    data=data.pivot(index=['AvailabilityZone','InstanceType'],columns=['ProductDescription'])
    data.columns=[x[1] for x in data.columns]
    data.drop('Windows',axis=1,inplace=True)
    res=pd.DataFrame(data['SUSE Linux']-data['Linux/UNIX'],columns=['SUSE_Linux_price_diff']).reset_index()
    res.to_csv('SUSE_Linux_price_diff.csv',index=False)

    # Filtering the dataset for OS-type = 'Linux/UNIX' only
    df_us = df_us[(df_us.ProductDescription == 'Linux/UNIX')].sort_values(by="Timestamp") 
    df_us.drop('ProductDescription', axis=1, inplace=True)  
    df_us=df_us.groupby(['AvailabilityZone', 'InstanceType']).apply(lambda x:x.set_index('Timestamp').resample('H').mean().fillna(method='ffill').fillna(method='bfill')).reset_index()

    sample_submission_us=df_us[['AvailabilityZone','InstanceType']]
    sample_submission_us['merge']=sample_submission_us.apply(lambda x:'_'.join(x.values),axis=1)
    sample_submission_us=pd.DataFrame(list(map(lambda x:x.split('_'),sample_submission_us['merge'].unique())),columns=['AvailabilityZone','InstanceType'])

    # save dict
    res=[]
    for i in df_us.columns:
           if 'object' in str(df_us[str(i)].dtype):
                res.append([i,[df_us[str(i)].values,df_us[str(i)].astype('category').cat.codes.values]])

    region_df=pd.DataFrame(list(zip(res[0][1][0],res[0][1][1])),columns=['AvailabilityZone','AvailabilityZone_label']).drop_duplicates().sort_values('AvailabilityZone_label')
    region_df.to_csv('region_label_dict_us.csv',index=False)

    ins_df=pd.DataFrame(list(zip(res[1][1][0],res[1][1][1])),columns=['InstanceType','InstanceType_label']).drop_duplicates().sort_values('InstanceType_label')
    ins_df.to_csv('ins_label_dict_us.csv',index=False)

    # Multiple column label encoding with dtype = object
    for i in df_us.columns:
           if 'object' in str(df_us[str(i)].dtype):
                df_us[str(i)]=df_us[str(i)].astype('category').cat.codes 

    df_us['Date'] = df_us['Timestamp'].dt.date
    fromDate = min(df_us['Timestamp'])
    df_us['Timedel'] = (df_us['Timestamp'] - fromDate).astype(np.int64)/100000000000
    df_us = df_us[['AvailabilityZone', 'InstanceType','Timestamp', 'Timedel', 'Date', 'SpotPrice']]

    # logging
    if logging: print(str(datetime.datetime.now()).split('.')[0],'train test split...')
    train_start = df_us.Date.min() + timedelta(1)
    train_end = train_start + timedelta(52)

    test_start = train_end + timedelta(1)
    test_end = test_start + timedelta(6)

    print ('Train set starts from: ', train_start)
    print ('Train set ends on: ', train_end)

    print ('Test set starts from: ', test_start)
    print ('Test set ends on: ', test_end)

    # Train Data
    mask = (df_us['Date'] >= train_start) & (df_us['Date'] <= train_end)
    train = df_us.loc[mask]
    train.drop('Date', axis = 1, inplace = True)
    train.reset_index(inplace = True, drop = True)

    # Test Data
    mask = (df_us['Date'] >= test_start) & (df_us['Date'] <= test_end)
    test = df_us.loc[mask]
    test.drop('Date', axis = 1, inplace = True)
    test.reset_index(inplace = True, drop = True)

    if logging: print(str(datetime.datetime.now()).split('.')[0],'output...')
    train.to_csv('train_test_train_us.csv',index=False)
    test.to_csv('train_test_test_us.csv',index=False)
