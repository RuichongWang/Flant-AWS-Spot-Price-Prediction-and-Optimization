import os
import warnings
from datetime import timedelta
import datetime
import numpy as np 
import pandas as pd

warnings.filterwarnings('ignore')

def pre_process(input_dir,logging=False):
    if logging: print(str(datetime.datetime.now()).split('.')[0],'load data...')
    files=os.listdir(input_dir)
    files=[x for x in files if '.txt' in x] # take txt files (spot price history data) only 

    # processing txt file to dataframe
    with open(input_dir+files[0]) as f:
        df_us=f.readlines()[5:]
    df_us=list(map(lambda x:x.split('|'),df_us))
    df_us=pd.DataFrame(df_us)
    new_header = df_us.iloc[0] 
    df_us = df_us[1:] 
    df_us.columns = [x.strip() for x in new_header] 
    df_us=df_us[['AvailabilityZone', 'InstanceType', 'ProductDescription','SpotPrice', 'Timestamp']]

    for file in files[1:]:
        with open(input_dir+file) as f:
            try:
                temp_df=f.readlines()[5:]
                temp_df=list(map(lambda x:x.split('|'),temp_df))
                temp_df=pd.DataFrame(temp_df)
                new_header = temp_df.iloc[0] 
                temp_df = temp_df[1:] 
                temp_df.columns = [x.strip() for x in new_header] 
                temp_df=temp_df[['AvailabilityZone', 'InstanceType', 'ProductDescription','SpotPrice', 'Timestamp']]    

                df_us = pd.concat((df_us,temp_df))
            except:
                print('Error: %s damaged!'%file)

    # clean columns
    df_us.dropna(inplace=True)
    df_us['Timestamp'] = pd.to_datetime(df_us['Timestamp'], format = '%Y-%m-%d %H:%M:%S.%f', utc=True)
    df_us = df_us.sort_values(by="Timestamp")
    df_us['AvailabilityZone']=df_us['AvailabilityZone'].str.strip()
    df_us['InstanceType']=df_us['InstanceType'].str.strip()
    df_us['ProductDescription']=df_us['ProductDescription'].str.strip()
    df_us['SpotPrice']=df_us['SpotPrice'].astype(float)

    # calculate price add on
    if logging: print(str(datetime.datetime.now()).split('.')[0],'calculating price add on ...')
    data=df_us.groupby(['AvailabilityZone','InstanceType','ProductDescription'],as_index=False)['SpotPrice'].min()
    data=data.pivot(index=['AvailabilityZone','InstanceType'],columns=['ProductDescription'])
    data.reset_index(inplace=True)
    data.columns=[x[1] if x[1]!='' else x[0] for x in data.columns]
    data['SUSE_Linux_price_diff']=data['SUSE Linux']-data['Linux/UNIX']
    data.to_csv('../data/SUSE_Linux_price_diff.csv',index=False)

    # Filtering the dataset for OS-type = 'Linux/UNIX' only
    df_us = df_us[(df_us.ProductDescription == 'Linux/UNIX')].sort_values(by="Timestamp") 
    df_us.drop('ProductDescription', axis=1, inplace=True)  
    df_us=df_us.groupby(['AvailabilityZone', 'InstanceType']).apply(lambda x:x.set_index('Timestamp').resample('H').mean().fillna(method='ffill').fillna(method='bfill')).reset_index()

    sample_submission_us=df_us[['AvailabilityZone','InstanceType']]
    sample_submission_us['merge']=sample_submission_us.apply(lambda x:'_'.join(x.values),axis=1)
    sample_submission_us=pd.DataFrame(list(map(lambda x:x.split('_'),sample_submission_us['merge'].unique())),columns=['AvailabilityZone','InstanceType'])

    # save original name of one hot encoding for region and instance type
    res=[]
    for i in df_us.columns:
           if 'object' in str(df_us[str(i)].dtype):
                res.append([i,[df_us[str(i)].values,df_us[str(i)].astype('category').cat.codes.values]])

    region_df=pd.DataFrame(list(zip(res[0][1][0],res[0][1][1])),columns=['AvailabilityZone','AvailabilityZone_label']).drop_duplicates().sort_values('AvailabilityZone_label')
    region_df.to_csv('../data/region_label_dict_us.csv',index=False)

    ins_df=pd.DataFrame(list(zip(res[1][1][0],res[1][1][1])),columns=['InstanceType','InstanceType_label']).drop_duplicates().sort_values('InstanceType_label')
    ins_df.to_csv('../data/ins_label_dict_us.csv',index=False)

    # Multiple column label encoding with dtype = object
    for i in df_us.columns:
           if 'object' in str(df_us[str(i)].dtype):
                df_us[str(i)]=df_us[str(i)].astype('category').cat.codes 

    # encoding date time 
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

    print('Train set starts from: ', train_start)
    print('Train set ends on: ', train_end)
    print('Test set starts from: ', test_start)
    print('Test set ends on: ', test_end)

    # save Train Data
    mask = (df_us['Date'] >= train_start) & (df_us['Date'] <= train_end)
    train = df_us.loc[mask]
    train.drop('Date', axis = 1, inplace = True)
    train.reset_index(inplace = True, drop = True)

    # save Test Data
    mask = (df_us['Date'] >= test_start) & (df_us['Date'] <= test_end)
    test = df_us.loc[mask]
    test.drop('Date', axis = 1, inplace = True)
    test.reset_index(inplace = True, drop = True)

    if logging: print(str(datetime.datetime.now()).split('.')[0],'output...')
    train.to_csv('../data/train_test_train_us.csv',index=False)
    test.to_csv('../data/train_test_test_us.csv',index=False)
